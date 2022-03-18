from tkinter import *
import tkinter
import numpy as np

padding = 5

def makebuttons(root, fields):
    entries = []
    for field in fields:
        row = tkinter.Frame(root)
        btn = tkinter.Button(master=row, text=field)
        row.pack(side=tkinter.RIGHT, padx=padding, pady=padding)
        btn.pack(side=tkinter.RIGHT)
        entries.append((field, btn))
    return entries

def makeformfields(root, fields):
    entries = []
    for field in fields:
        row = tkinter.Frame(root)
        lab = tkinter.Label(row, width=15, text=field, anchor='w')
        ent = tkinter.Entry(row)
        row.pack(side=tkinter.TOP, padx=padding, pady=padding)
        lab.pack(side=tkinter.LEFT)
        ent.pack(side=tkinter.RIGHT, expand=tkinter.YES)
        entries.append((field, ent))
    return entries

def makespinbox(root, field, _from, _to):
    row = tkinter.Frame(root)
    lab = tkinter.Label(row, width=15, text=field, anchor='w')
    ent = tkinter.Spinbox(row,from_=_from, to=_to)
    row.pack(side=tkinter.TOP, padx=padding, pady=padding)
    lab.pack(side=tkinter.LEFT)
    ent.pack(side=tkinter.RIGHT, expand=tkinter.YES)
    return ent
