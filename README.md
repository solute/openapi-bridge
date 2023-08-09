# openapi-bridge

OpenAPI endpoint decorator with pydantic (>=2.0.0) integration.

Allows for almost seamless integration of pydantic models with OpenAPI,
generating YAML for the endpoint from type hints.


## Using the `@endpoint` decorator

Example:

```python
@endpoint("/foo/{foo_id}")
def foo(*, foo_id: int, debug: Optional[bool] = False) -> pydantic_models.Foo:
   """
       Foo summary.

       Lengthy foo description.

       @param foo_id The Foo ID in the database.
       @param debug Set to True to include debug information.
   """
   result = _do_something(id, debug)
   return pydantic_models.Foo(**result)
```

As you can see from the example, the decorator takes a path (which may include
a path parameter, in this case `id`). You can also give it an HTTP method, a
path prefix (e.g. to distinguish between internal and external API functions),
and security directives.

Information about the endpoint is gathered from both the type annotations of
the decorated function and its docstring.

(!) Every parameter (except the optional `user`) must be keyword-only, have a
type hint and a @param help text in the docstring. Un-annotated or undocumented
parameters are considered to be a hard error and will raise an exception on
startup.

Normally you can just return an instance of the annotated type, and the
decorator will handle it correctly, adding an HTTP status 200. If you need to
return something else, e.g. some redirect or a 204 ("no content"), you can to
return a `(raw_content, http_status)` tuple instead, e.g.:

```python
return None, 204
```

The docstring of an endpoint contains its summary, description, the section the
documentation is listed under, and parameter help, as well as (optionally) its
response in various circumstances.

The summary is the first paragraph of the docstring; the description is taken
to be any further paragraphs until the first @keyword.

We recognize the following keywords to designate parts of the documentation:
 - `@section <section name>`: endpoint is listed in this section.
 - `@param <param name> <help text>`: explanation of the given parameter.
 - `@example <param name> <example text>`: example values of the parameter.
 - `@response <http status> <JSON>`: allows for non-standard responses.


## YAML Generation

If you're building a Connexion app, you can use the collected endpoints in your
`create_app()` function:

```python
def create_app():
    # TODO: import all modules with @endpoints here!!
    api_specs = {
        "paths": openapi_bridge.PATHS["default"],
        **openapi_bridge.get_pydantic_schemata(pydantic_models),
    }
    connexion_app.add_api(api_specs)
```

(!) You need to import all the modules with endpoints here in order to register
them. This is easy to forget! If you test a new endpoint and only ever get 404,
you might have forgotten to import that module ;-)
