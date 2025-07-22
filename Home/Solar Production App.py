from kivy.lang import Builder
from kivymd.app import MDApp
from pvlib.location import Location
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivy.uix.boxlayout import BoxLayout
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import pvlib
from kivymd.uix.textfield import MDTextField
from kivy.uix.image import Image
from io import BytesIO

KV = '''
BoxLayout:
    orientation: 'vertical'

    MDTopAppBar:
        title: 'Solar Production Model'
        elevation: 10

    MDRaisedButton:
        text: 'Select Date'
        pos_hint: {'center_x': 0.5}
        on_release: app.show_date_input()

    MDLabel:
        id: date_label
        text: 'No date selected'
        halign: 'center'

    MDLabel:
        text: 'Panel Tilt (degrees)'
        halign: 'center'

    MDSlider:
        id: tilt_slider
        min: 0
        max: 45
        value: 20
        step: 1
        on_value: app.update_tilt(self.value)

    MDLabel:
        id: tilt_label
        text: 'Tilt: 20°'
        halign: 'center'

    MDRaisedButton:
        text: 'Generate Graph'
        pos_hint: {'center_x': 0.5}
        on_release: app.generate_graph()

    BoxLayout:
        id: graph_box
        size_hint_y: 3
'''

class SolarApp(MDApp):
    location = (33.0485, 35.4884)
    date = None
    tilt = 20
    dialog = None

    def build(self):
        return Builder.load_string(KV)

    def show_date_input(self):
        if not self.dialog:
            self.dialog = MDDialog(
                title='Enter Date (YYYY-MM-DD):',
                type='custom',
                content_cls=MDTextField(hint_text='2025-07-22'),
                buttons=[
                    MDFlatButton(text='CANCEL', on_release=self.close_dialog),
                    MDFlatButton(text='OK', on_release=self.set_date)
                ]
            )
        self.dialog.open()

    def close_dialog(self, *args):
        self.dialog.dismiss()

    def set_date(self, *args):
        date_input = self.dialog.content_cls.text
        try:
            self.date = datetime.strptime(date_input, '%Y-%m-%d').date()
            self.root.ids.date_label.text = f'Selected Date: {self.date}'
            self.close_dialog()
        except ValueError:
            # self.root.ids.date_label.text = 'Invalid date format! Use YYYY-MM-DD.'
            self.date = datetime.strptime("2025-07-22", '%Y-%m-%d').date()
            self.root.ids.date_label.text = f'Selected Date: {self.date}'
            self.close_dialog()

    def update_tilt(self, value):
        self.tilt = value
        self.root.ids.tilt_label.text = f'Tilt: {int(value)}°'

    def generate_graph(self):
        if not self.date:
            self.root.ids.date_label.text = 'Please select a date first.'
            return

        times = pd.date_range(
            start=datetime.combine(self.date, datetime.min.time()),
            end=datetime.combine(self.date, datetime.max.time()),
            freq='5min', tz='Asia/Jerusalem'
        )

        solpos = pvlib.solarposition.get_solarposition(times, self.location[0], self.location[1])
        tus = Location(self.location[0], self.location[1], 'Asia/Jerusalem', 666)
        clear_sky_estimates = pvlib.location.Location.get_clearsky(tus, times=times)
        ghi = clear_sky_estimates['ghi']
        dhi = clear_sky_estimates['dhi']
        dni = clear_sky_estimates['dni']

        # tus = Location(self.location[0], self.location[1], 'Asia/Jerusalem', 666)
        # ghi = pvlib.clearsky.ineichen(times, tus)
        # ghi = pvlib.clearsky.ineichen(times, self.location[0], self.location[1])['ghi']
        # dhi = ghi * 0.1  # Rough estimate of diffuse horizontal irradiance
        # dni = ghi * 0.9  # Rough estimate of direct normal irradiance

        poa_irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt,
            surface_azimuth=180,
            solar_zenith=solpos['apparent_zenith'],
            solar_azimuth=solpos['azimuth'],
            dni=dni,
            ghi=ghi,
            dhi=dhi
        )['poa_global']

        panel_capacity = 32 * 615 / 1000  # kW
        inverter_limit = 15  # kW

        theoretical_power = poa_irradiance / 1000 * panel_capacity
        capped_power = np.minimum(theoretical_power, inverter_limit)

        plt.clf()
        plt.plot(times, theoretical_power, label='Theoretical Max (kW)')
        plt.plot(times, capped_power, label='Inverter Limited (15kW)')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Power (kW)')
        plt.title('Solar Production Throughout the Day')
        plt.gcf().autofmt_xdate()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = buf.read()
        buf.close()

        image = Image()
        image.texture = Image(BytesIO(img_data)).texture

        self.root.ids.graph_box.clear_widgets()
        self.root.ids.graph_box.add_widget(image)

if __name__ == '__main__':
    SolarApp().run()
