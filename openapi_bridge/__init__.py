# pylint: disable=c-extension-no-member, too-many-locals, too-many-branches, no-else-return
import ast
import decimal
import datetime
import enum
import functools
import inspect
import re
import textwrap
import typing
from itertools import zip_longest

import pydantic


PATHS = {}


def get_pydantic_schemata(pydantic_model_module):
    """
    Generate OpenAPI #/components/schemas dict for all pydantic models in the
    given module.

    Usage:

        import solute.synapi.pydantic_models as _pydantic_models
        ...
        schemas = solute_openapi_bridge.get_pydantic_schemata(_pydantic_models)
    """
    schemata = []
    for symbol in sorted(dir(pydantic_model_module)):
        if symbol[0] != symbol[0].upper():
            continue  # skip non-uppercase symbols
        if symbol == "BaseModel":
            continue
        attr = getattr(pydantic_model_module, symbol)
        if getattr(attr, "__nodocs__", False):
            continue
        if isinstance(attr, pydantic.main.ModelMetaclass):
            schemata.append(attr)
    schemata = pydantic.schema.schema(schemata)
    _patch_dict(schemata, "title", None)
    _patch_dict(schemata, "$ref", _definitions_to_schemas)
    return {"components": {"schemas": schemata["definitions"]}}


def _patch_dict(d, key, transform):
    if isinstance(d, dict):
        for k, v in list(d.items()):  # so we can pop
            if isinstance(v, (dict, list)):
                _patch_dict(v, key, transform)
            elif k == key:
                if transform:
                    d[k] = transform(v)
                else:
                    d.pop(k)
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, dict):
                _patch_dict(v, key, transform)


def _definitions_to_schemas(ref):
    return ref.replace("definitions/", "components/schemas/")


class endpoint:
    '''
    OpenAPI endpoint decorator with pydantic integration.

    Allows for almost seamless integration of pydantic models with OpenAPI,
    generating YAML for the endpoint from type hints.


    # Collecting Endpoint Information

    Example:

        @endpoint("/foo/{foo_id}")
        def foo(user, *, foo_id: int, debug: Optional[bool] = False) -> _pydantic_models.Foo:
           """
               Foo summary.

               Lengthy foo description.

               @param foo_id The Foo ID in the database.
               @param debug Set to True to include debug information.
           """
           result = _do_something(id, debug)
           return _pydantic_models.Foo(result)

    As you can see from the example, the decorator takes a path (which may include
    a path parameter, in this case `id`). You can also give it an HTTP method, a
    path prefix (e.g. to distinguish between internal and external API functions),
    and security directives.

    Information about the endpoint is gathered from both the type annotations of
    the decorated function and its docstring.

    (!) Every parameter except `user` must be keyword-only, have a type hint and a
    @param help text in the docstring. Un-annotated or undocumented parameters are
    considered to be a hard error and will raise an exception on startup.

    Normally you can just return an instance of the annotated type, and the
    decorator will handle it correctly, adding an HTTP status 200. If you need to
    return something else, e.g. some redirect or a 204 ("no content"), you can to
    return a `(raw_content, http_status)` tuple instead, e.g.:

        return None, 204

    The docstring of an endpoint contains its summary, description, the section the
    documentation is listed under, and parameter help, as well as (optionally) its
    response in various circumstances.

    The summary is the first paragraph of the docstring; the description is taken
    to be any further paragraphs until the first @keyword.

    We recognize the following keywords to designate parts of the documentation:
     - @section <section name>: endpoint is listed in this section.
     - @param <param name> <help text>: explanation of the given parameter.
     - @example <param name> <example text>: example values of the parameter.
     - @response <http status> <JSON>: allows for non-standard responses.


    # YAML Generation

    If you're building a Connexion app, you can use the collected endpoints in your
    `create_app()` function:

        def create_app():
            from . import (
                baseproducts,
                brands,
                categories,
                offers,
                products,
                shops,
            )  # pylint: disable=unused-import
            ...
            api_specs = _yaml.load(
                yaml_file.read_text(encoding="utf-8"),
                Loader=_yaml.CLoader,
            )
            ...
            from solute_openapi_bridge import PATHS
            api_specs["paths"] = {
                **PATHS["default"],
                **PATHS.get(api_spec["servers"][0]["url"], {}),
            }
            connexion_app.add_api(api_specs)

    (!) You need to import all the modules with endpoints here in order to register
    them. This is easy to forget! If you test a new endpoint and only ever get 404,
    you might have forgotten to import that module ;-)

    You can then just put the PATHS in the YAML specs and make them known to Connexion.
    '''

    SECURITY_NONE = {}
    SECURITY_BASIC = {"security": [{"basic": []}]}

    def __init__(
        self,
        path,
        method="get",
        path_prefix="default",
        security=None,
        response_model_exclude_none=False,
    ):
        self.path = path
        self.method = method.lower()
        self.path_prefix = path_prefix
        self.security = security if security is not None else endpoint.SECURITY_BASIC
        self.response_model_exclude_none = response_model_exclude_none

    def __call__(self, fn):
        parameters = []
        request_body = None
        summary, description, docs = _parse_docstring(fn.__doc__)
        fqname = f"{fn.__module__}.{fn.__name__}"
        spec = inspect.getfullargspec(fn)
        returns = spec.annotations["return"]
        assert not spec.varargs, "*args is not supported on endpoints (because they can't be annotated)"
        assert not spec.varkw, "**kwargs is not supported on endpoints (because they can't be annotated)"
        assert set(spec.args) <= {"user"}, "all params except 'user' must be keyword-only"  # magic connexion params
        assert docs["response"].get("200") or issubclass(returns, pydantic.BaseModel), "if you don't return a pydantic model, you need to document the @response 200"
        return_reference = f"#/components/schemas/{returns.__name__}"
        for arg in spec.kwonlyargs:
            annotation = spec.annotations[arg]
            in_ = "path" if f"{{{arg}}}" in self.path else "query"
            assert arg in docs["param"], f"Undescribed parameter in {fqname}: {arg}"
            param = {
                "name": arg,
                "in": in_,
                "required": not str(annotation).startswith("typing.Optional"),
                "description": docs["param"][arg],
                "schema": self._get_schema(
                    annotation,
                    (spec.kwonlydefaults or {}).get(arg),
                    in_,
                    example=docs["example"].get(arg),
                    name=arg,
                ),
            }
            if "type" in param["schema"]:
                if param["schema"]["type"] == "array" and param["in"] == "path":
                    param["explode"] = False
                    param["style"] = "simple"

                elif param["schema"]["type"] == "object":
                    param["explode"] = True
                    param["style"] = "deepObject"

                elif "$ref" in param["schema"]:
                    param["explode"] = True
                    param["style"] = "deepObject"

            if arg == "body" and self.method == "post":
                request_body = {
                    "content": {
                        "application/json": {"schema": param["schema"]},
                    },
                    "description": docs["param"][arg],
                }
            elif annotation in (bytes, typing.Optional[bytes]) and self.method == "post":
                if not request_body:
                    request_body = {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                        },
                    }
                request_body["content"]["multipart/form-data"]["schema"]["properties"][arg] = {"description": docs["param"][arg], **param["schema"]}
            else:
                parameters.append(param)
        PATHS.setdefault(self.path_prefix, {}).setdefault(self.path, {})[
            self.method
        ] = {
            "parameters": parameters,
            "summary": summary,
            "description": description,
            "tags": list(docs["section"]),
            "operationId": fqname,
            "responses": {
                200: _response("OK", "application/json", return_reference),
                400: _response(
                    "Client error",
                    "application/problem+json",
                    "#/components/schemas/Problem",
                ),
                **_custom_responses(docs["response"]),
            },
            **self.security,
            **({"requestBody": request_body} if request_body else {}),
        }

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, returns):
                if isinstance(result, pydantic.BaseModel):
                    return result.dict(by_alias=True, exclude_none=self.response_model_exclude_none), 200
                else:
                    return result, 200
            return result

        return wrapper

    def _get_schema(self, annotation, default, in_, name=None, example=None):
        origin = typing.get_origin(annotation)
        if str(annotation).startswith("typing.Optional"):
            inner_type, _ = typing.get_args(annotation)
            return self._get_schema(
                inner_type, default, in_, name=name, example=example
            )
        if annotation is str:
            result = {"type": "string"}
        elif annotation is bytes:
            result = {
                "type": "string",
                "format": "binary",
            }
        elif annotation is int:
            result = {"type": "integer"}
        elif annotation is bool:
            result = {"type": "boolean"}
        elif annotation is float:
            result = {"type": "number"}
        elif annotation is decimal.Decimal:
            result = {"type": "number", "format": "decimal"}
        elif annotation is datetime.datetime:
            result = {
                "type": "string",
                "format": "date-time",
            }
        elif annotation is datetime.date:
            result = {
                "type": "string",
                "format": "date",
            }
        elif origin is list:
            result = {
                "type": "array",
                "items": self._get_schema(
                    typing.get_args(annotation)[0],
                    default=None,
                    in_="query",
                    name=name,
                    example=None,
                ),
            }
            if in_ == "path":
                result["minItems"] = 1
                result["maxItems"] = 100
        elif origin is typing.Literal:
            choices = typing.get_args(annotation)
            [py_type] = {type(choice) for choice in choices}
            result = {
                "type": self._get_schema(py_type, default=None, in_="query")["type"],
                "enum": list(choices),
            }
        elif isinstance(annotation, type):
            if issubclass(annotation, pydantic.types.ConstrainedList):
                min_items, max_items = annotation.min_items, annotation.max_items
                result = {
                    "type": "array",
                    "items": self._get_schema(
                        annotation.item_type,
                        default=None,
                        in_="query",
                        name=name,
                        example=None,
                    ),
                }
                if min_items is not None:
                    result["minItems"] = min_items
                if max_items is not None:
                    result["maxItems"] = max_items
            elif issubclass(annotation, pydantic.types.ConstrainedInt):
                le, ge = annotation.le, annotation.ge
                result = {
                    "type": "integer",
                }
                if le is not None:
                    result["maximum"] = le
                if ge is not None:
                    result["minimum"] = ge
            elif issubclass(annotation, pydantic.types.ConstrainedStr):
                result = {
                    "type": "string",
                    "pattern": annotation.regex.pattern,
                }
            elif issubclass(annotation, enum.Enum):
                choices = list(annotation.__members__)
                result = {
                    "type": "string",
                    "enum": list(choices),
                }
            elif issubclass(annotation, pydantic.BaseModel):
                # result = {"$ref": f"#/components/schemas/{annotation.__name__}"}
                result = {
                    "type": "object",
                    "properties": {
                        name: self._get_schema(field.annotation, field.default, "query")
                        for name, field in annotation.__fields__.items()
                    },
                }
                if annotation.Config.extra.value == "allow":
                    result["additionalProperties"] = self._get_schema(
                        getattr(annotation, "__extra_type__", str),
                        None,
                        "query",
                    )
            else:
                raise ValueError(f"Unknown type {annotation} of parameter {name}")
        else:
            raise ValueError(f"Unknown type {annotation} of parameter {name}")
        if default is not None:
            result["default"] = default
        if example:
            result["example"] = ast.literal_eval(example)
        return result


def _parse_docstring(docstring):
    # parse docstring according to rules of class `endpoint`.
    # example:
    """
    My summary.

    My lengthy description

    @section MySection
    @param foo Give foo value here.
    @example foo 42
    @response 200 {"description: "Foo", ...}
    ----
    Private part 8===D
    """
    docstring, *_private_docs = re.split(
        r"^\s*-{4,}$", docstring, flags=re.MULTILINE, maxsplit=1
    )
    summary, *parts = re.split(r"@(\w+)", docstring)
    assert summary.startswith("\n"), "Docstrings need to break after opening quotes"
    summary = textwrap.dedent(summary)
    summary, _, description = summary.partition("\n\n")
    iterator = iter(parts)
    docs = {"param": {}, "response": {}, "section": {}, "example": {}}
    for key, value in zip_longest(iterator, iterator, fillvalue=None):
        assert key in docs, f"invalid @{key} in docstring, allowed are: {sorted(docs)}"
        if " " in value.strip():
            name, text = re.split(r"\s", value.strip(), maxsplit=1)
            text = textwrap.dedent(text).strip()
        else:
            name = value.strip()
            text = None
        docs[key][name] = text
    return summary.strip(), description.strip(), docs


def _response(description, mime_type, schema):
    return {
        "description": description,
        "content": {mime_type: {"schema": {"$ref": schema}}},
    }


def _custom_responses(responses):
    # responses: dict(status code -> JSON response)
    result = {}
    for status, rest in responses.items():
        status = int(status)
        result[status] = ast.literal_eval(rest)
    return result
