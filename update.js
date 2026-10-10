module.exports = {
  run: [
    // the launcher itself
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // the official Iris-3B code
    {
      method: "shell.run",
      params: {
        path: "app/iris-3b",
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install --no-deps -e iris-3b"
        ]
      }
    }
  ]
}
