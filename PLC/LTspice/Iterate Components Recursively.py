class Component:
    def __init__(self, name, nominal_value, tolerance):
        self.name = name
        self.nominal_value = nominal_value
        self.tolerance = tolerance
        self.nominal_with_tolerance_high = nominal_value * (1 + tolerance / 100)
        self.nominal_with_tolerance_low = nominal_value * (1 - tolerance / 100)


class SpecialComponent(Component):
    def __init__(self, name, nominal_value, tolerance, tolerance_p, tolerance_n):
        super().__init__(name, nominal_value, tolerance)
        self.tolerance_p = tolerance_p
        self.tolerance_n = tolerance_n
        self.nominal_with_tolerance_high_p = self.nominal_with_tolerance_high * (1 + tolerance_p / 100)
        self.nominal_with_tolerance_high_n = self.nominal_with_tolerance_high * (1 - tolerance_n / 100)
        self.nominal_with_tolerance_low_p = self.nominal_with_tolerance_low * (1 + tolerance_p / 100)
        self.nominal_with_tolerance_low_n = self.nominal_with_tolerance_low * (1 - tolerance_n / 100)


all_components = []             # Add all components to here
run_index = 0                   # Count the number of runs
current_configuration = []      # List of strings for text adjustment
all_components.append(Component("Rs", nominal_value=1000, tolerance=20))
all_components.append(SpecialComponent("Ls", nominal_value=10e-6, tolerance=5, tolerance_p=2, tolerance_n=3))
all_components.append(SpecialComponent("Cs", nominal_value=100e-3, tolerance=1, tolerance_p=1, tolerance_n=1))


def recursive_adjustment(components, depth=0):
    global run_index
    global current_configuration
    if not components:
        run_index += 1
        print(f"{run_index = }")
        print("\n".join(current_configuration) + "\n\n")
        return

    current_component = components[0]
    for attr, value in current_component.__dict__.items():
        if attr in ["name", "tolerance", "tolerance_p", "tolerance_n"]:
            continue
        current_configuration = current_configuration[:depth]
        current_configuration.append(f'{current_component.name = }, {attr = }, {value = }')
        recursive_adjustment(components[1:], depth + 1)


recursive_adjustment(all_components)
print(f"Finished! number of runs = {run_index}")
