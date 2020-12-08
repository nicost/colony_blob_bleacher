# colony_blob_bleacher Conda instructions

Conda environment files are provided to easily replicate the Python environment to run this code (and keep environments in sync).  To use these, downoad and install miniconda from: https://docs.conda.io/en/latest/miniconda.html.  

Then run the following command:

`conda env create -f env_os_independent.yml`

On Windows, use `env_win.yml` as the `.yml` file instead.

Then run:

`conda activate blob_bleacher`

and you should be set to go.  

In Pycharm, you need to select the correct conda environment. Go to `File > Settings >Project: colony_blob_bleacher > Python Interpreter`.  Click on the gear icon on the right (next to Python Interpreter) and click `Ad`".  Now select `Conda Environment` on the left, select `Existing environment`, and locate the Python executable in the environment you just created (for me it is in C:\Users\Nico\anaconda3\envs\blob_bleacher\python.exe)
