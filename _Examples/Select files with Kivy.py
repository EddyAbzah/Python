from kivy.app import App
from kivy_garden.filebrowser import FileBrowser


class TestApp(App):
    def build(self):
        browser = FileBrowser(select_string='Select')
        browser.bind(on_success=self._fbrowser_success, on_canceled=self._fbrowser_canceled)
        return browser

    def _fbrowser_canceled(self, instance):
        print('cancelled, Close self.')

    def _fbrowser_success(self, instance):
        if len(instance.selection) > 0:
            print(f'{instance.selection = }')
        else:
            print(f'{instance.path = }')

TestApp().run()