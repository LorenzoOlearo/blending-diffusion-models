# Blending Stable Diffusion

## How to run
1. Install the requirements
```bash
pip install -r requirements.txt
```

Once the requirements are satisfied, in order to run the code you have to create
a configuration file following the structure of the `sample-config.json` file
and pass it as an argument to the main script.

2. Run the code
```bash
python src/main.py <config_file>
```

When the code is executed, the results will be saved in the `out/` directory
with the following structure:
```bash
out/
`-- {prompt_1}-BLEND-{prompt_2}
    `-- {seed} 
        |-- [{prompt_1}-BLEND-{prompt_2}]-[from_{t}]-[to_{t}]-[{scheduler}]-[{model_id}]-[p1_{t}]-[p2_{t}]
        |   |-- blending-{prompt_1}-BLEND-{prompt_2}.png
        |   |-- config.json
        |   |-- denoising-{prompt_1}-BLEND-{prompt_2}.gif
        |   |-- final_image-{prompt_2}.png
        |   |-- final_image-{prompt_1}-BLEND-{prompt_2}.png
        |   |-- final_image-{prompt_1}.png
        |   `-- intermediate-{prompt_1}-timestep-{t}.png
```
Along with the results, the configuration file used to generate the results will
be saved in the same directory in order to keep track of the parameters used.