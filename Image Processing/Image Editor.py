import os
from fnmatch import fnmatch
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.config import Config


Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Window.size = (1000, 800)
gui_spacing = 20
gui_input_size_hint_x = 0.86
gui_input_size_hint_y = [3.5, 0.1, 3.5, 3.5, 0.1, 1.5, 1.5]
gui_others_size_hint_x = [0.7, 0.125]
gui_others_size_hint_y = 1
timestamp_default = "%Y-%m-%d %H-%M-%S"
sider_min = 0
sider_max = 2
sider_default = 1


def edit_photo(image_path, output_path, brightness_factor=1.0, contrast_factor=1.0, saturation_factor=1.0, sharpness_factor=1.0, sharpness_enhancement=False, color_balance=(1.0, 1.0, 1.0)):
    image = Image.open(image_path)
    exif = image.getexif()

    if brightness_factor != 1:
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    if contrast_factor != 1:
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    if saturation_factor != 1:
        image = ImageEnhance.Color(image).enhance(saturation_factor)
    if sharpness_factor != 1:
        image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)
    if sharpness_enhancement:
        image = image.filter(ImageFilter.UnsharpMask())
    if any([n != 1 for n in color_balance]):
        r, g, b = image.split()
        r = r.point(lambda i: i * color_balance[0])
        g = g.point(lambda i: i * color_balance[1])
        b = b.point(lambda i: i * color_balance[2])
        image = Image.merge('RGB', (r, g, b))

    image.save(output_path, exif=exif)


def get_files(folders, include_subfolders, filter_in, filter_out):
    files = []
    for folder in folders:
        folder = folder.strip()
        for root, dirs, all_files in os.walk(folder):
            for file in all_files:
                if (filter_in == "" or any(fnmatch(file, f) for f in filter_in)) and (filter_out == "" or not any(fnmatch(file, f) for f in filter_out)):
                    files.append(os.path.join(root, file))
            if not include_subfolders:
                break
    return files


class PhotoEditorApp(App):
    edit_counter = 1
    edit_name_memory = f"edit {edit_counter:02}"

    def build(self):
        main_layout = BoxLayout(orientation='vertical', spacing=gui_spacing)

        # Inputs:
        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[0]))
        horizontal_layout.add_widget(Label(text="Folder", size_hint=(gui_input_size_hint_x, 1)))
        self.folder_input = TextInput(text="")
        horizontal_layout.add_widget(self.folder_input)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[1]))
        horizontal_layout.add_widget(Label(text="Include subfolders", size_hint=(gui_input_size_hint_x, 1)))
        self.include_subfolders_input = CheckBox(active=False)
        horizontal_layout.add_widget(self.include_subfolders_input)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[2]))
        horizontal_layout.add_widget(Label(text="Filter in", size_hint=(gui_input_size_hint_x, 1)))
        self.filter_in_input = TextInput(text="*.jpg")
        horizontal_layout.add_widget(self.filter_in_input)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[3]))
        horizontal_layout.add_widget(Label(text="Filter out", size_hint=(gui_input_size_hint_x, 1)))
        self.filter_out_input = TextInput(text="")
        horizontal_layout.add_widget(self.filter_out_input)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[4]))
        horizontal_layout.add_widget(Label(text="Export log (saved in the first folder)", size_hint=(gui_input_size_hint_x, 1)))
        self.export_log_input = CheckBox(active=False)
        horizontal_layout.add_widget(self.export_log_input)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[5]))
        horizontal_layout.add_widget(Label(text="Edit Name (if empty the original will be overwritten!)", size_hint=(gui_others_size_hint_x[0], 1)))
        self.edit_name_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.edit_name_reset.bind(on_release=self.edit_name_input_reset)
        horizontal_layout.add_widget(self.edit_name_reset)
        self.edit_name_input = TextInput(text=self.edit_name_memory)
        horizontal_layout.add_widget(self.edit_name_input)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_input_size_hint_y[6]))
        horizontal_layout.add_widget(Label(text="Timestamp (leave empty to omit)", size_hint=(gui_others_size_hint_x[0], 1)))
        self.timestamp_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.timestamp_reset.bind(on_release=self.timestamp_input_reset)
        horizontal_layout.add_widget(self.timestamp_reset)
        self.timestamp_input = TextInput(text=timestamp_default)
        horizontal_layout.add_widget(self.timestamp_input)
        main_layout.add_widget(horizontal_layout)

        # Sliders:
        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.brightness_label = Label(text=f"Brightness = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.brightness_label)
        self.brightness_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.brightness_reset.bind(on_release=self.on_brightness_slider_change)
        horizontal_layout.add_widget(self.brightness_reset)
        self.brightness_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.brightness_slider.bind(value=self.on_brightness_slider_change)
        horizontal_layout.add_widget(self.brightness_slider)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.contrast_label = Label(text=f"Contrast = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.contrast_label)
        self.contrast_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.contrast_reset.bind(on_release=self.on_contrast_slider_change)
        horizontal_layout.add_widget(self.contrast_reset)
        self.contrast_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.contrast_slider.bind(value=self.on_contrast_slider_change)
        horizontal_layout.add_widget(self.contrast_slider)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.saturation_label = Label(text=f"Saturation = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.saturation_label)
        self.saturation_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.saturation_reset.bind(on_release=self.on_saturation_slider_change)
        horizontal_layout.add_widget(self.saturation_reset)
        self.saturation_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.saturation_slider.bind(value=self.on_saturation_slider_change)
        horizontal_layout.add_widget(self.saturation_slider)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.sharpness_label = Label(text=f"Sharpness = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.sharpness_label)
        self.sharpness_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.sharpness_reset.bind(on_release=self.on_sharpness_slider_change)
        horizontal_layout.add_widget(self.sharpness_reset)
        self.sharpness_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.sharpness_slider.bind(value=self.on_sharpness_slider_change)
        horizontal_layout.add_widget(self.sharpness_slider)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.sharpness_enhancement_label = Label(text=f"Sharpness enhancement = {False}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.sharpness_enhancement_label)
        self.sharpness_enhancement_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.sharpness_enhancement_reset.bind(on_release=self.on_sharpness_enhancement_bool_change)
        horizontal_layout.add_widget(self.sharpness_enhancement_reset)
        self.sharpness_enhancement_bool = CheckBox()
        self.sharpness_enhancement_bool.bind(active=self.on_sharpness_enhancement_bool_change)
        horizontal_layout.add_widget(self.sharpness_enhancement_bool)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.color_balance_r_label = Label(text=f"Color Balance (Red) = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.color_balance_r_label)
        self.color_balance_r_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.color_balance_r_reset.bind(on_release=self.on_color_balance_r_slider_change)
        horizontal_layout.add_widget(self.color_balance_r_reset)
        self.color_balance_r_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.color_balance_r_slider.bind(value=self.on_color_balance_r_slider_change)
        horizontal_layout.add_widget(self.color_balance_r_slider)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.color_balance_g_label = Label(text=f"Color Balance (Green) = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.color_balance_g_label)
        self.color_balance_g_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.color_balance_g_reset.bind(on_release=self.on_color_balance_g_slider_change)
        horizontal_layout.add_widget(self.color_balance_g_reset)
        self.color_balance_g_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.color_balance_g_slider.bind(value=self.on_color_balance_g_slider_change)
        horizontal_layout.add_widget(self.color_balance_g_slider)
        main_layout.add_widget(horizontal_layout)

        horizontal_layout = BoxLayout(orientation='horizontal', spacing=gui_spacing, size_hint=(1, gui_others_size_hint_y))
        self.color_balance_b_label = Label(text=f"Color Balance (Blue) = {sider_default:.2f}", size_hint=(gui_others_size_hint_x[0], 1))
        horizontal_layout.add_widget(self.color_balance_b_label)
        self.color_balance_b_reset = Button(text="Reset", size_hint=(gui_others_size_hint_x[1], 1))
        self.color_balance_b_reset.bind(on_release=self.on_color_balance_b_slider_change)
        horizontal_layout.add_widget(self.color_balance_b_reset)
        self.color_balance_b_slider = Slider(min=sider_min, max=sider_max, value=sider_default, orientation='horizontal')
        self.color_balance_b_slider.bind(value=self.on_color_balance_b_slider_change)
        horizontal_layout.add_widget(self.color_balance_b_slider)
        main_layout.add_widget(horizontal_layout)

        # Last but not least:
        edit_button = Button(text="Start editing", size_hint=(0.25, 3), pos_hint={"x": 0.5 - (0.25 / 2)})
        edit_button.bind(on_release=self.start)
        main_layout.add_widget(edit_button)
        return main_layout

    def edit_name_input_reset(self, instance):
        self.edit_name_input.text = self.edit_name_memory

    def timestamp_input_reset(self, instance):
        self.timestamp_input.text = timestamp_default

    def on_brightness_slider_change(self, instance, value=sider_default):
        self.brightness_slider.value = value
        self.brightness_label.text = f"Brightness = {value:.2f}"

    def on_contrast_slider_change(self, instance, value=sider_default):
        self.contrast_slider.value = value
        self.contrast_label.text = f"Contrast = {value:.2f}"

    def on_saturation_slider_change(self, instance, value=sider_default):
        self.saturation_slider.value = value
        self.saturation_label.text = f"Saturation = {value:.2f}"

    def on_sharpness_slider_change(self, instance, value=sider_default):
        self.sharpness_slider.value = value
        self.sharpness_label.text = f"Sharpness = {value:.2f}"

    def on_sharpness_enhancement_bool_change(self, instance, value=False):
        self.sharpness_enhancement_bool.active = value
        self.sharpness_enhancement_label.text = f"Sharpness enhancement = {value}"

    def on_color_balance_r_slider_change(self, instance, value=sider_default):
        self.color_balance_r_slider.value = value
        self.color_balance_r_label.text = f"Color Balance (Red) = {value:.2f}"

    def on_color_balance_g_slider_change(self, instance, value=sider_default):
        self.color_balance_g_slider.value = value
        self.color_balance_g_label.text = f"Color Balance (Green) = {value:.2f}"

    def on_color_balance_b_slider_change(self, instance, value=sider_default):
        self.color_balance_b_slider.value = value
        self.color_balance_b_label.text = f"Color Balance (Blue) = {value:.2f}"

    def start(self, instance):
        folders = self.folder_input.text.replace('\n', ',').replace(';', ',').split(',')
        include_subfolders = self.include_subfolders_input.active
        filter_in = self.filter_in_input.text.replace('\n', ',').replace(';', ',').split(',')
        filter_out = self.filter_out_input.text.replace('\n', ',').replace(';', ',').split(',')
        timestamp = datetime.now().strftime(self.timestamp_input.text)
        edit_name = self.edit_name_input.text
        self.edit_name_memory = edit_name

        brightness_factor = self.brightness_slider.value
        contrast_factor = self.contrast_slider.value
        saturation_factor = self.saturation_slider.value
        sharpness_factor = self.sharpness_slider.value
        sharpness_enhancement = self.sharpness_enhancement_bool.active
        color_balance = (self.color_balance_r_slider.value, self.color_balance_g_slider.value, self.color_balance_b_slider.value)

        string_applied_enhancements = f"""These are the applied enhancements:
{brightness_factor = :.2f}
{contrast_factor = :.2f}
{saturation_factor = :.2f}
{sharpness_factor = :.2f}
{sharpness_enhancement = :.2f}
color_balance = ({color_balance[0]:.2f}, {color_balance[1]:.2f}, {color_balance[2]:.2f})"""

        try:
            files = get_files(folders, include_subfolders, filter_in, filter_out)
            if len(files) > 0:
                for index_file, file in enumerate(files):
                    if edit_name != "":
                        file_out = f" ({edit_name}).".join(file.rsplit('.', 1))
                    else:
                        file_out = file
                    if timestamp != "":
                        file_out = f" _ {timestamp})".join(file_out.rsplit(')', 1))
                    edit_photo(file, file_out, brightness_factor, contrast_factor, saturation_factor, sharpness_factor, sharpness_enhancement, color_balance)
            else:
                raise Exception("There are no files to edit")

            if self.export_log_input.active:
                if edit_name != "":
                    file_out = f"{folders[0]}\\Python Image Editor ({edit_name}).txt"
                else:
                    file_out = f"{folders[0]}\\Python Image Editor.txt"
                if timestamp != "":
                    file_out = f" _ {timestamp})".join(file_out.rsplit(')', 1))
                f = open(file_out, "a")
                f.write(f"Python Image Editor\n\n")
                f.write(f"edit_name = {edit_name}\n")
                f.write(f"Date = {timestamp}\n")
                f.write(f"Files:\n")
                for index_file, file in enumerate(files):
                    f.write(f"File {index_file + 1:03} = {file}\n")
                f.write(f"\n{string_applied_enhancements}\n")
                f.close()

            self.edit_counter += 1
            self.edit_name_input.text = f"edit {self.edit_counter:02}"

            self.show_popup(f"Successfully edited {len(files)} file(s) in folder(s):\n'{'\n'.join(folders)}'", string_applied_enhancements)
        except Exception as e:
            self.show_popup("Error", f"An error occurred: {str(e)}")

    @staticmethod
    def show_popup(title, content):
        popup = Popup(title=title, content=Label(text=content), size_hint=(None, None), size=(400, 400))
        popup.open()


if __name__ == "__main__":
    PhotoEditorApp().run()
