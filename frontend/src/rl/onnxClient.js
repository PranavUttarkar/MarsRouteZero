/**
 * Optional browser policy (PRD §6.4). After `python -m rl.export_onnx`, serve `mars_policy.onnx` from /public.
 * Install: npm install onnxruntime-web
 */
export async function loadOnnxPolicySession() {
  try {
    const ort = await import("onnxruntime-web");
    return await ort.InferenceSession.create("/mars_policy.onnx", {
      executionProviders: ["wasm"],
    });
  } catch {
    return null;
  }
}
