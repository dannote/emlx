Mix.install([
  {:bumblebee, "~> 0.6.0"},
  {:nx, path: "/Users/valente/coding/nx/nx", override: true},
  {:emlx, path: "/Users/valente/coding/emlx"},
])

Nx.global_default_backend({EMLX.Backend, device: :gpu})
Nx.Defn.default_options(compiler: EMLX)

repo_id = "CompVis/stable-diffusion-v1-4"
opts = [params_variant: "fp16", type: :bf16, backend: EMLX.Backend]

{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "openai/clip-vit-large-patch14"})
{:ok, clip} = Bumblebee.load_model({:hf, repo_id, subdir: "text_encoder"}, opts)
{:ok, unet} = Bumblebee.load_model({:hf, repo_id, subdir: "unet"}, opts)
{:ok, vae} = Bumblebee.load_model({:hf, repo_id, subdir: "vae"}, [architecture: :decoder] ++ opts)
{:ok, scheduler} = Bumblebee.load_scheduler({:hf, repo_id, subdir: "scheduler"})
{:ok, featurizer} = Bumblebee.load_featurizer({:hf, repo_id, subdir: "feature_extractor"})
{:ok, safety_checker} = Bumblebee.load_model({:hf, repo_id, subdir: "safety_checker"}, opts)

serving =
  Bumblebee.Diffusion.StableDiffusion.text_to_image(clip, unet, vae, tokenizer, scheduler,
    num_steps: 20,
    num_images_per_prompt: 1,
    safety_checker: safety_checker,
    safety_checker_featurizer: featurizer,
    compile: [batch_size: 1, sequence_length: 60],
    # Option 1
    defn_options: [compiler: EMLX]
    # Option 2 (reduces GPU usage, but runs noticeably slower)
    # Also remove `backend: EXLA.Backend` from the loading options above
    # defn_options: [compiler: EXLA, lazy_transfers: :always]
  )

Nx.Serving.start_link(name: StableDiffusion, serving: serving)

prompt = "numbat, forest, high quality, detailed, digital art"
negative_prompt = "darkness, rainy, foggy"

IO.puts("Running run")
:timer.tc(fn ->
  Nx.Serving.run(serving, %{prompt: prompt, negative_prompt: negative_prompt})
end)
|> IO.inspect(label: "run")

IO.puts("Running batched_run")
Task.async(fn ->
  :timer.tc(fn ->
    Nx.Serving.batched_run(StableDiffusion, %{prompt: prompt, negative_prompt: negative_prompt})
  end)
end)
|> Task.await(45_000)
|> IO.inspect(label: "batched_run")
