# README

Injection_check.py file detects and prevents prompt injection, hateful content, and prompt leaking when generating text using GPT-2. It uses the 'transformers' library.

## Requirements

- Anaconda Navigator
- Python 3.6 or higher

## Setup

1. Open Anaconda Navigator and create a new environment:
   - Click on "Environments" on the left side menu.
   - Click on "Create" at the bottom.
   - Name your environment, e.g., "gpt2_check".
   - Choose the appropriate Python version (3.9.16 is the one I used).

2. Install the required packages for this environment:
    - notebook ((Your can install notebook through Anaconda packages))
    - transformers (Your can install transformers through Anaconda packages)
    - openai (You can install openai by running *pip install openai* in the terminal)

3. Back to the home page and launch jupyter Notebook on this environment

4. Create a new Python3 Notebook named "prompt_hacking_defense.py"

5. Copy the code in prompt_hacking_defense.py from this repo to the newly created prompt_hacking_defense.py file.

6. Replace *openai.api_key = "YOUR OPEN AI KEY"* with your openAI api key in check_input_for_legality and check_output_for_legality functions. 
   - You can find your openAI api key here: https://platform.openai.com/account/api-keys

## Running the script

1. Open Anaconda Navigator and launch jupyter Notebook for "gpt2_check" environment

2. Navigate to the prompt_hacking_defense.py file and open it

3. Once the file is loaded, click on the Run button from the navigation menu.
