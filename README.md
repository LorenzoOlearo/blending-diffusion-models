# Blending Diffusion Models

This is the repository for the soon to be published paper "How to Blend Concepts
in Diffusion Models".

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

When the code is executed, all the blending methods in the `blend_methods` array
of the config file will be executed with all of the seeds in the `seeds` array.
The results will be saved in the `out/<prompt_1>-BLEND-<prompt_2>` 
Along with the results, the configuration file used to generate the results will
be saved in the same directory in order to keep track of the parameters used.

WORK IN PROGRESS - For any questions, feel free to contact me.