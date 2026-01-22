import requests
import os
from pprint import pprint
from pydantic import BaseModel, Field
from argparse import ArgumentParser
from importlib import import_module
from os.path import basename, splitext


def main():
    try:
        api_key = "sk-3xIkP3EQpqG7qioiokFXRQIwSUCwwT2OuLZOHPOzb3w"
    except KeyError:
        raise ValueError("LANGFLOW_API_KEY environment variable not found. Please set your API key in the environment variables.")
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key  # Authentication key from environment variable
    }


    arg_parser = ArgumentParser()
    arg_parser.add_argument("type", type=int, help="1: Multiple Choice, 2: True or False, 3: Discursive")
    arg_parser.add_argument("input", type=str, help="Input text for the API")
    arg_parser.add_argument("-s", "--session_id", type=str, help="Session ID for the API request", default="test_session")
    arg_parser.add_argument("-f", "--flow_name", type=str, help="name of the flow to be used in URL", default='q_generator_json')
    arg_parser.add_argument("-m", "--model", type=str, help="path to a file with an output_model pydantic model", default='./model.py')
    args = arg_parser.parse_args()


    output_model_module = splitext(basename(args.model))[0]
    output_model_dict = getattr(import_module(output_model_module, args.model), 'output_model')
    output_model, question_type = output_model_dict.get(args.type)

    url = f"http://localhost:7860/api/v1/run/{args.flow_name}"

    payload = {
        "output_type": "chat",
        "input_type": "text",
        "tweaks":{
            "TextInput-TeYlu":{"input_value": args.input},
            "TextInput-ikFP4":{"input_value": question_type}
            },
        "session_id": args.session_id
    }
    
    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes
    
        # Print response
        data = response.json().get('outputs')[0].get('outputs')[0].get('messages')[0].get('message')
        output = output_model.model_validate_json(data)
        pprint(output.model_dump())
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

if __name__ == "__main__":
    main()
