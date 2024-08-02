import json


def get_example(data, path=""):
    examples = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, (dict, list)):
                examples.update(get_example(value, new_path))
            else:
                examples[new_path] = value

    elif isinstance(data, list):
        if data:
            examples.update(get_example(data[0], path))
        else:
            examples[path] = "[]"

    return examples


def main(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    examples = get_example(data)

    for key, value in examples.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    json_file_path = "C:\\Users\\a926163\\OneDrive - Eviden\\Escritorio\\Github\\datasets\\coco_densepose\\archive\\coco2014\\annotations\\densepose_coco_2014_train.json"  # Replace with your JSON file path
    main(json_file_path)
