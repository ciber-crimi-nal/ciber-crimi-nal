import utils
from dni import *
from datetime import datetime
import unicodedata
import re
import unicodedata
import re
import string


def normalize_dni_data(dni_data: dict[str, str]) -> dict[str, str]:
    format_data = dni_data.copy()

    for element in dni_data.items():
        if element[1] is not None:
            value = element[1].upper()
            value = utils.delete_chars(value, "/")
            value = utils.delete_chars(value, "—")
            value = utils.delete_chars(value, "-")
            value = utils.delete_chars(value, ",")
            value = utils.delete_chars(value, ".")
            value = utils.delete_chars(value, ".")
            value = utils.delete_chars(value, "+")
            value = utils.delete_chars(value, "”")
            value = utils.delete_chars(value, "*")
            value = utils.delete_chars(value, "'")
            value = utils.delete_chars(value, "_")
            value = utils.delete_chars(value, "|")
            value = utils.delete_chars(value, "!")

            value = str.replace(value, "?", "2")
            value = value.strip()

            if element[0] in [EMISION, VALIDEZ, NACIMIENTO]:
                value = utils.validate_re(value, date_re)
            elif element[0] == NUM_DNI:
                value = utils.validate_re(value, dni_re)
            elif element[0] == SEXO:
                value = utils.validate_re(value, sexo_re)
            elif element[0] == NACIONALIDAD:
                value = utils.validate_re(value, nacionalidad_re)
            elif element[0] == NUM_SOPORTE:
                value = value[0:3].replace("0", "O") + value[3:]
                value = utils.validate_re(value, num_soporte_re)
            elif element[0] == CAN:
                value = utils.validate_re(value, can_re)
            elif element[0] == EQUIPO:
                value = utils.validate_re(value, equipo_re)
        else:
            value = ""

        format_data[element[0]] = value

    return format_data
