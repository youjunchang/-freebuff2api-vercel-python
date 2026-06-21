import uvicorn

from freebuff2api.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "freebuff2api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
