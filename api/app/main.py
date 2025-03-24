from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import status, prediction, information
from app.services.model_service import load_model

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    contact=settings.CONTACT,
    license_info=settings.LICENSE_INFO,
    openapi_tags=settings.OPENAPI_TAGS,
    docs_url=None,
    redoc_url=None,
    version=settings.APP_VERSION,
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(status.router)
app.include_router(prediction.router)
app.include_router(information.router)

# Fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentation API",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1.3/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.1.3/swagger-ui.css",
        swagger_ui_parameters={
            "customCss": """
                .topbar-wrapper img[alt="Swagger UI"] {
                    content: url("/static/logo.png");
                    height: 40px;
                    width: auto;
                }
                .swagger-ui .topbar {
                    background-color: #9c1d1d;
                }
            """
        }
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Documentation ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# Fonction custom pour personnaliser OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Correction pour la version d'OpenAPI
    openapi_schema["openapi"] = "3.0.2"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Événement de démarrage
@app.on_event("startup")
async def startup_event():
    load_model()