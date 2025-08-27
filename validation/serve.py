import uvicorn
from server.app import app, args


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
