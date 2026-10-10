// Removes the Python environment and the cloned code. Model weights stay in the
// Hugging Face cache and generated images stay in app/outputs.
module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "app/iris-3b"
      }
    }
  ]
}
