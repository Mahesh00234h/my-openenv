from fastapi import FastAPI, HTTPException, Query, Response
from env import ChiefOfStaffEnv

app = FastAPI(title="Email Triage Environment")

env = ChiefOfStaffEnv()


@app.get("/")
def root():
    return {"name": "email-triage-env", "status": "running", "endpoints": ["/reset", "/step", "/state", "/docs"]}


@app.api_route("/reset", methods=["GET", "POST"])
def reset(task_id: str = Query(default="easy_cos")) -> Response:
    import json
    try:
        obs = env.reset(task_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(content=json.dumps(obs, ensure_ascii=False), media_type="application/json; charset=utf-8")


@app.post("/step")
async def step(action: dict):
    return env.step(action)


@app.get("/state")
def state() -> Response:
    import json
    return Response(content=json.dumps(env.state(), ensure_ascii=False), media_type="application/json; charset=utf-8")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
