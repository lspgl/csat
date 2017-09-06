class StyleSheet:
    # CSS-like Stylesheet class for QT style persistence
    # withough having to re-enter the whole sheet at every change

    def __init__(self, style_str):
        # Convert Style string from CSS to Python dict
        self.style = {s.split(':')[0]: s.split(':')[1] for s in style_str.split(';')[:-1]}

    def get(self):
        # Return dict as CSS string
        style_str = ''
        for key in self.style:
            style_str += key + ':' + str(self.style[key]) + ';\n'
        style_str = style_str[:-2]
        return style_str

    def set(self, **kwargs):
        # Change or append style to dict
        # CSS keywords as arguments with _ for -
        for key, value in iter(kwargs.items()):
            key = key.replace('_', '-')
            self.style[key] = str(value)


def main():
    style = ("background-color: #ff000f;" +
             "color: red;" +
             "border-bottom:5px;")

    s = StyleSheet(style)
    s.set(background_color='blue')
    print(s.get())


if __name__ == '__main__':
    main()
