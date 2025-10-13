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

import annotated_types
import pydantic
import pydantic_core

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
    schemata = {}
    for symbol in sorted(dir(pydantic_model_module)):
        if symbol == "BaseModel" or symbol == "RootModel":
            continue
        if symbol[0] != symbol[0].upper():
            continue  # skip non-uppercase symbols
        if symbol[0] == "_":  # skip private symbols
            continue
        attr = getattr(pydantic_model_module, symbol)
        if getattr(attr, "__nodocs__", False):  # do not include in specs
            continue
        if isinstance(attr, pydantic._internal._model_construction.ModelMetaclass):
            if attr is pydantic.main.BaseModel:
                continue  # pure base models are not allowed
            schema = attr.model_json_schema()
            if schema.get("$defs") and symbol in schema["$defs"]:
                schema = schema["$defs"][symbol]
            else:
                schema.pop("$defs", None)
            _patch_dict(schema, "title", None)
            _patch_dict(schema, "force", None)
            _patch_dict(schema, "default", None, filter_null=True)
            _patch_dict(schema, "$ref", _definitions_to_schemas)
            _remove_null_type(schema)
            _flatten_single(schema, "anyOf")
            _replace_const(schema)
            schemata[symbol] = schema
            continue
        if str(attr) == "<enum 'Enum'>":
            continue  # pure enums are not allowed
        if str(attr).startswith("<enum '"):
            adapter = pydantic.TypeAdapter(attr)
            schema = adapter.json_schema()
            if "description" not in schema:
                schema["description"] = "An enumeration."
            _patch_dict(schema, "title", None)
            schemata[symbol] = schema
            continue
    return {"components": {"schemas": schemata}}


def _flatten_single(d, key):
    if isinstance(d, dict):
        items = list(d.items())
        for k, v in items:
            if isinstance(v, (list, dict)) and key in v:
                if isinstance(v[key], dict):
                    d[k].update(v[key])
                elif len(v[key]) == 1:
                    d[k].update(v[key][0])
                else:
                    continue
                d[k].pop(key, None)

            elif isinstance(v, (dict, list)):
                _flatten_single(v, key)
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, dict):
                _flatten_single(v, key)


def _replace_const(d):
    if isinstance(d, dict):
        items = list(d.items())
        for k, v in items:
            if k == "const":
                d["enum"] = [v]
                d.pop("const")
            elif isinstance(v, (dict, list)):
                _replace_const(v)
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, dict):
                _replace_const(v)


def _patch_dict(d, key, transform, filter_null=False):
    if isinstance(d, dict):  # noqa:PLR1702 (too-many-nested-blocks)
        for k, v in list(d.items()):  # so we can pop
            if isinstance(v, (dict, list)):
                _patch_dict(v, key, transform)
            elif k == key:
                if transform:
                    d[k] = transform(v)
                else:
                    if filter_null:
                        if not d[k]:
                            d.pop(k)
                    else:
                        d.pop(k)
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, dict):
                _patch_dict(v, key, transform)


def _definitions_to_schemas(ref):
    return ref.replace("definitions/", "components/schemas/").replace(
        "$defs/", "components/schemas/"
    )


def _remove_null_type(d):
    if isinstance(d, dict):
        for k, v in list(d.items()):  # so we can pop
            if isinstance(v, dict):
                if v == {"type": "null"}:
                    d.pop(k)
                else:
                    _remove_null_type(v)
            if isinstance(v, list):
                v = [i for i in v if i != {"type": "null"}]
                _remove_null_type(v)
                d[k] = v
    elif isinstance(d, list):
        for v in d:
            if isinstance(v, dict):
                _remove_null_type(v)


class endpoint:
    '''
    OpenAPI endpoint decorator with pydantic integration.

    Allows for almost seamless integration of pydantic models with OpenAPI,
    generating YAML for the endpoint from type hints.

    # Parameters

    - method: HTTP verb, defaults to "get"
    - path_prefix: The optional prefix, to support multiple APIs (e.g. internal vs. external).
    - security: A custom security, defaults to basic auth
    - response_model_exclude_none: Whether to remove Nones from the response (defaults to False)
    - deprecated: Whether to mark this endpoint as deprecated

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
        *,
        method="get",
        path_prefix="default",
        security=None,
        response_model_exclude_none=False,
        deprecated=False,
    ):
        self.path = path
        self.method = method.lower()
        self.path_prefix = path_prefix
        self.security = security if security is not None else endpoint.SECURITY_BASIC
        self.response_model_exclude_none = response_model_exclude_none
        self.deprecated = deprecated

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
            if arg.startswith("_"):
                # skip underscore-prefixed arguments — those are private
                continue
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

            if arg == "body" and self.method in ("post", "patch", "put"):
                request_body = {
                    "content": {
                        "application/json": {"schema": param["schema"]},
                    },
                    "description": docs["param"][arg],
                }
            elif annotation in (bytes, typing.Optional[bytes]) and self.method in ("post", "patch", "put"):
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
        endpoint_config = {
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
        if self.deprecated:
            endpoint_config["deprecated"] = True
        PATHS.setdefault(self.path_prefix, {}).setdefault(self.path, {})[
            self.method
        ] = endpoint_config

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, returns):
                if isinstance(result, pydantic.BaseModel):
                    return result.model_dump(by_alias=True, exclude_none=self.response_model_exclude_none), 200
                else:
                    return result, 200
            return result

        return wrapper

    def _get_schema(self, annotation, default, in_, name=None, example=None):  # noqa:PLR0912 (too-many-branches)
        constraints = None
        origin = typing.get_origin(annotation)
        if str(annotation).startswith("typing.Optional"):
            inner_type, _ = typing.get_args(annotation)
            return self._get_schema(
                inner_type, default, in_, name=name, example=example
            )
        elif str(annotation).startswith("typing.Annotated"):
            args = typing.get_args(annotation)
            assert len(args) == 2
            annotation = args[0]
            constraints = args[1]
            origin = typing.get_origin(annotation)

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
            types = {type(choice) for choice in choices}
            assert len(types) == 1, f"Literal must have all the same types, but got {types}"
            result = {
                "type": self._get_schema(types.pop(), default=None, in_="query")["type"],
                "enum": list(choices),
            }
        elif isinstance(annotation, type):
            if issubclass(annotation, enum.Enum):
                choices = list(annotation.__members__)
                result = {
                    "type": "string",
                    "enum": list(choices),
                }
            elif issubclass(annotation, pydantic.BaseModel):
                result = {
                    "type": "object",
                    "properties": {
                        name: self._get_schema(field.annotation, field.default, "query")
                        for name, field in annotation.model_fields.items()
                    },
                }
                if annotation.model_config.get("extra") == "allow":
                    result["additionalProperties"] = self._get_schema(
                        getattr(annotation, "__extra_type__", str),
                        None,
                        "query",
                    )
            else:
                raise ValueError(f"Unknown type {annotation} of parameter {name}")
        else:
            raise ValueError(f"Unknown annotation {annotation} of parameter {name}")

        # add constraints if present
        if constraints is None:
            pass
        elif isinstance(constraints, pydantic.types.StringConstraints):
            result["pattern"] = constraints.pattern
        elif isinstance(constraints, pydantic.fields.FieldInfo):
            for m in constraints.metadata:
                if isinstance(m, annotated_types.Ge):
                    result["minimum"] = m.ge
                elif isinstance(m, annotated_types.Le):
                    result["maximum"] = m.le
                elif isinstance(m, annotated_types.MaxLen):
                    if origin is list:
                        result["maxItems"] = m.max_length
                    else:
                        result["maxLength"] = m.max_length
                elif isinstance(m, annotated_types.MinLen):
                    if origin is list:
                        result["minItems"] = m.min_length
                    else:
                        result["minLength"] = m.min_length
                else:
                    raise ValueError(f"Unknown constraint {m} of parameter {name}")
        else:
            raise ValueError(f"Unknown constraints type {type(constraints)} of parameter {name}")

        # add default value if present
        if default is not None and default is not pydantic_core.PydanticUndefined and default != "":
            result["default"] = default

        # add example if present
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
