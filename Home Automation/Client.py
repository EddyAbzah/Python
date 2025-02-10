import requests


SUCCESS_STATUS_CODE = 200
FAIL_STATUS_CODE = 400
FAIL_STATUS_CODE_BAD = 401


def create_device(device_number: int):
    res = requests.post(f'http://localhost:8000/devices/{device_number}')
    return res


def delete_device(device_number: int):
    res = requests.delete(f'http://localhost:8000/devices/{device_number}')
    return res


def get_all_params(device_number: int):
    res = requests.get(f'http://localhost:8000/devices/{device_number}/params')
    return res


def test_create():
    random_add = 10
    delete_device(random_add)
    result = create_device(random_add)
    assert result.text == '"Success"'
    print("result: ", result.text)


def test_exist():
    random_add = 10
    result = create_device(random_add)
    assert result.text == '{"detail":"Device already exists"}'
    print("result: ", result.text)
