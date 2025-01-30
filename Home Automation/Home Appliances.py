import time
import threading
import builtins         # to access the original print function
import functools        # to preserve the metadata of the original function

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        # Call the original print function
        builtins.print(*args, **kwargs)


# Override built-in print with our safe version
print = safe_print


def log_function_call(func):
    """Decorator: Logs function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] Calling {func.__name__} with args: {args[1:]}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper


class HomeAppliance:
    def __init__(self, name):
        self.__is_on = False
        self.name = name
        self.lock = threading.Lock()

    @log_function_call
    def turn_on(self, delay):
        with self.lock:
            if not self.__is_on:
                print(f"Waiting {delay} seconds for the device to turn on")
                time.sleep(delay)
                self.__is_on = True
                print(f"{self.name} is now ON.")
            else:
                print(f"{self.name} is already ON.")

    @log_function_call
    def turn_off(self):
        with self.lock:
            if self.__is_on:
                self.__is_on = False
                print(f"{self.name} is now OFF.")
            else:
                print(f"{self.name} is already OFF.")

    def is_on(self):
        return self.__is_on


class SmartBulb(HomeAppliance):
    def __init__(self, name="SmartBulb"):
        super().__init__(name)
        self.__color = "White"

    def __str__(self):
        # "self.is_on()" because there is no access to parent class "self.__is_on"
        return f"Bulb is {f'ON, Color: {self.__color}' if self.is_on() else 'OFF'}"

    @log_function_call
    def set_color(self, color):
        with self.lock:
            if self.is_on():
                self.__color = color
                print(f"{self.name} color set to {color}.")
            else:
                print("Turn on the bulb first!")


class SmartTV(HomeAppliance):
    def __init__(self, name="SmartTV"):
        super().__init__(name)
        self.__channel = 1

    def __str__(self):
        # "self.is_on()" because there is no access to parent class "self.__is_on"
        return f"TV is {f'ON, Channel: {self.__channel}' if self.is_on() else 'OFF'}"

    @log_function_call
    def set_channel(self, channel):
        with self.lock:
            if self.is_on():
                self.__channel = channel
                print(f"{self.name} changed to channel {channel}.")
            else:
                print("Turn on the TV first!")


if __name__ == '__main__':
    bulb = SmartBulb()
    tv = SmartTV()

    thread1 = threading.Thread(target=bulb.turn_on, args=(0.1, ))   # args expects a tuple
    thread2 = threading.Thread(target=bulb.set_color, args=("Blue", ))
    thread3 = threading.Thread(target=tv.turn_on, args=(4,))
    thread4 = threading.Thread(target=tv.set_channel, args=(5,))

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    print("\nFinal State:")
    print(bulb)
    print(tv)
