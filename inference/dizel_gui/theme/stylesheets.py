# dizel_ui/theme/stylesheets.py

from .colors import resolve

def get_button_style(bg_color, hover_color, text_color, radius=16, border_color=None):
    """Generate a standard rounded button stylesheet."""
    bg = resolve(bg_color)
    hover = resolve(hover_color)
    text = resolve(text_color)
    
    border_css = f"border: 1px solid {resolve(border_color)};" if border_color else "border: none;"
    
    return f"""
        QPushButton {{
            background-color: {bg};
            color: {text};
            border-radius: {radius}px;
            {border_css}
            padding: 4px 12px;
        }}
        QPushButton:hover {{
            background-color: {hover};
        }}
        QPushButton:pressed {{
            background-color: {hover};
            margin-top: 1px;
            margin-left: 1px;
        }}
    """

def get_frame_style(bg_color, radius=0, border_color=None):
    bg = resolve(bg_color)
    border_css = f"border: 1px solid {resolve(border_color)};" if border_color else "border: none;"
    
    return f"""
        QFrame {{
            background-color: {bg};
            border-radius: {radius}px;
            {border_css}
        }}
    """

def get_scrollbar_style(bg_color, handle_color, hover_color):
    bg = resolve(bg_color)
    handle = resolve(handle_color)
    hover = resolve(hover_color)
    
    return f"""
        QScrollBar:vertical {{
            border: none;
            background: {bg};
            width: 10px;
            margin: 0px 0px 0px 0px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {handle};
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {hover};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
    """

def get_input_style(bg_color, border_color, text_color, focus_color, radius=4):
    bg = resolve(bg_color)
    border = resolve(border_color)
    text = resolve(text_color)
    focus = resolve(focus_color)
    return f'''
        QLineEdit {{
            background-color: {bg};
            border: 1px solid {border};
            color: {text};
            border-radius: {radius}px;
        }}
        QLineEdit:focus {{
            border: 1px solid {focus};
        }}
    '''
