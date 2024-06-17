import binaryninja

from .binaryninja_actions import *

binaryninja.PluginCommand.register(
    "Dump", "Dump a current loaded binary for QueryX.", dump
)
