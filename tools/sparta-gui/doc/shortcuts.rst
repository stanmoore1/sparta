******************
Keyboard Shortcuts
******************

.. index:: keyboard shortcuts
.. index:: hotkeys
.. index:: keybindings

Almost all functionality is accessible from the menu of the editor
window or through keyboard shortcuts.  The following shortcuts are
available (On macOS use the Command key instead of Ctrl/Control).

.. list-table::
   :header-rows: 1
   :widths: 16 19 13 16 13 22

   * - Shortcut
     - Function
     - Shortcut
     - Function
     - Shortcut
     - Function
   * - Ctrl+N
     - New File
     - Ctrl+Z
     - Undo edit
     - Ctrl+Enter
     - Run Input
   * - Ctrl+O
     - Open File
     - Ctrl+Shift+Z
     - Redo edit
     - Ctrl+/
     - Stop Active Run
   * - Ctrl+K
     - Check Input
     -
     -
     -
     -
   * - Ctrl+Shift+F
     - View Text File
     - Ctrl+C
     - Copy text
     - Ctrl+Shift+V
     - Set Variables
   * - Ctrl+S
     - Save File
     - Ctrl+X
     - Cut text
     - Ctrl+I
     - Snapshot Image
   * - Ctrl+Shift+S
     - Save File As
     - Ctrl+V
     - Paste text
     - Ctrl+L
     - Slide Show
   * - Ctrl+Q
     - Quit Application
     - Ctrl+A
     - Select All
     - Ctrl+F
     - Find and Replace
   * - Ctrl+W
     - Close Window
     - TAB
     - Reformat line
     - Shift+TAB
     - Show Completions
   * - Ctrl+Shift+Enter
     - Run File
     - Ctrl+Shift+W
     - Show Variables
     - Ctrl+P
     - Preferences
   * - Ctrl+Shift+A
     - About GUI
     - Ctrl+Shift+H
     - Quick Help
     - Ctrl+Shift+G
     - SPARTA-GUI Docs
   * - Ctrl+Shift+R
     - Inspect Restart File
     - Ctrl+Shift+L
     - Output Window
     - Ctrl+Shift+C
     - Charts Window
   * - Ctrl+Shift+I
     - Image Window
     - Ctrl+Shift+M
     - SPARTA Manual
     - Ctrl+?
     - Context Help
   * - Ctrl+Shift+J
     - View Image or Movie File(s)
     - Ctrl+Shift+D
     - Plot Data File
     - Ctrl+Home / Ctrl+End
     - Go to Start / End
   * - Ctrl+Shift+T
     - Import Surface (STL / SPARTA)
     - Ctrl+Shift+E
     - Export to ParaView
     - Ctrl+E
     - Extend Run
   * - Ctrl+Shift+3
     - 3D Snapshot (VTK)
     - Ctrl+Shift+U
     - Check for SPARTA Update
     - Ctrl+1 … Ctrl+3
     - Run / Analyze / Visualize workspace
   * - Alt+Home
     - Welcome Screen
     - Ctrl+Shift+P
     - Command Palette
     - F1
     - Keyboard Shortcuts

Further keybindings of the editor window `are documented with the Qt
documentation
<https://doc.qt.io/qt-6/qplaintextedit.html#editing-key-bindings>`_.  In
case of conflicts the list above takes precedence.

All other windows only support a subset of keyboard shortcuts listed
above.  Typically, the shortcuts `Ctrl-/` (Stop Run), `Ctrl-W` (Close
Window), and `Ctrl-Q` (Quit Application) are supported.  Some sub-windows
also rebind shortcuts for window-specific actions:

- *Output* window: `Ctrl-N` jumps to the next warning or error,
  `Ctrl-S` saves the captured log to a file, `Ctrl-Y` exports any
  embedded YAML data, `Ctrl-Enter` runs the current input buffer.
- *Viewer* panel, whichever tab is in front: `Ctrl-S` saves the picture
  to a file and `Ctrl-C` copies it to the clipboard.  Both belong to the
  panel rather than to one of its viewers, so they mean the same thing on
  the snapshot, the slide show and the 3D scene.
- *Snapshot* tab: `Ctrl-D` copies the ``dump image`` / ``dump_modify``
  commands to the clipboard, `F9` shows or hides the settings sidebar,
  `F8` re-renders at the size of the panel, `Ctrl-W` closes the panel.
  `Ctrl-W` acts on the panel rather than on the editor tab -- which of
  the two a keystroke reaches depends on where the focus is, so the
  entry is named *Close Panel* to say so.
- *Sequence* tab: `Ctrl-E` exports the image sequence to a movie file.
