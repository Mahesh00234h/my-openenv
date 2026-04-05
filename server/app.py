from fastapi import FastAPI, HTTPException, Query, Response
from email_triage_env.env import EmailTriageEnv
from email_triage_env.models import Action

app = FastAPI(title="Email Triage Environment")

env = EmailTriageEnv()


@app.get("/")
def root():
    return {"name": "email-triage-env", "status": "running", "endpoints": ["/reset", "/step", "/state", "/docs"]}


@app.api_route("/reset", methods=["GET", "POST"])
def reset(task_id: str = Query(default="easy_triage")) -> Response:
    try:
        obs = env.reset(task_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(content=obs.model_dump_json(), media_type="application/json")


@app.post("/step")
def step(action: Action) -> Response:
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    import json
    payload = {
        "observation": json.loads(obs.model_dump_json()),
        "reward": json.loads(reward.model_dump_json()),
        "done": done,
        "info": info,
    }
    return Response(content=json.dumps(payload), media_type="application/json")


@app.get("/state")
def state() -> Response:
    import json
    return Response(content=json.dumps(env.state()), media_type="application/json")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
