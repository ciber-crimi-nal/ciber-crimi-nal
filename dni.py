import re

NUM_DNI = "num_dni"
NOMBRE = "nombre"
APELLIDO1 = "apellido1"
APELLIDO2 = "apellido2"
SEXO = "sexo"
NACIONALIDAD = "nacionalidad"
NUM_SOPORTE = "num_soporte"
EMISION = "emision"
VALIDEZ = "validez"
NACIMIENTO = "nacimiento"
CAN = "can"
DOMICILIO = "domicilio"
MUNICIPIO = "municipio"
PROVINCIA = "provincia"
MUNICIPIO_N = "municipio_nacimiento"
PROVINCIA_N = "provincia_nacimiento"
PROGENITORES = "progenitores"
PROGENITOR_1 = "progenitor_1"
PROGENITOR_2 = "progenitor_2"
EQUIPO = "equipo"
MRZ = "mrz"
FRONTAL = "front"
TRASERA = "back"
TIPO_DOC = "tipo_doc"
PAIS = "pais"
HASH_1 = "hash1"
HASH_2 = "hash2"
HASH_3 = "hash3"
FINAL_HASH = "final_hash"
OPT_DATA_1 = "opt_data1"
OPT_DATA_2 = "opt_data2"
APELLIDOS = "apellidos"


# (x1, y1, x2, y2)
POSICIONES = {
    NOMBRE: (0.38, 0.429, 0.703, 0.501),
    APELLIDO1: (0.38, 0.292, 0.703, 0.35),
    APELLIDO2: (0.38, 0.345, 0.703, 0.411),
    NUM_DNI: (0.441, 0.166, 0.749, 0.271),
    SEXO: (0.397, 0.53, 0.45, 0.592),
    NACIONALIDAD: (0.525, 0.528, 0.679, 0.591),
    NUM_SOPORTE: (0.392, 0.707, 0.585, 0.77),
    EMISION: (0.389, 0.615, 0.592, 0.682),
    VALIDEZ: (0.592, 0.615, 0.793, 0.682),
    NACIMIENTO: (0.775, 0.524, 0.985, 0.596),
    CAN: (0.778, 0.819, 0.99, 0.92),
    DOMICILIO: (0.27, 0.07, 0.95, 0.13),
    MUNICIPIO: (0.27, 0.123, 0.95, 0.18),
    PROVINCIA: (0.27, 0.171, 0.95, 0.228),
    MUNICIPIO_N: (0.27, 0.35, 0.9, 0.4),
    PROVINCIA_N: (0.27, 0.39, 0.9, 0.452),
    PROGENITORES: (0.27, 0.52, 0.9, 0.6),
    EQUIPO: (0.035, 0.268, 0.08, 0.57),
    MRZ: (0, 0.65, 1, 0.938),
}

ELEMENTOS_FRONTALES = [NOMBRE, APELLIDO1, APELLIDO2, NUM_DNI, SEXO, NACIONALIDAD, NUM_SOPORTE, EMISION, VALIDEZ,
                       NACIMIENTO, CAN]
ELEMENTOS_TRASEROS = [DOMICILIO, MUNICIPIO, PROVINCIA, MUNICIPIO_N, PROVINCIA_N, PROGENITORES, EQUIPO, MRZ]

date_re = re.compile("[0-9]{2} [0-9]{2} [0-9]{4}")
dni_re = re.compile("[0-9]{8}[A-HJ-NP-TV-Z]")
sexo_re = re.compile("[MF]")
nacionalidad_re = re.compile("[A-Z]{3}")
num_soporte_re = re.compile("[A-Z]{3}[0-9]{6}")
can_re = re.compile("[0-9]{6}")
equipo_re = re.compile("(0[1-9]|[1-4][0-9]|5[0-2])[0-9]{3}[A-Z][0-9A-Z][A-Z][0-9A-Z]")

char_table = ["T", "R", "W", "A", "G", "M", "Y", "F", "P", "D", "X", "B", "N", "J", "Z", "S", "Q", "V", "H", "L",
              "C", "K", "E"]


def get_dni() -> dict[str, str]:
    return {
        NOMBRE: "",
        APELLIDO1: "",
        APELLIDO2: "",
        NUM_DNI: "",
        SEXO: "",
        NACIONALIDAD: "",
        NUM_SOPORTE: "",
        EMISION: "",
        VALIDEZ: "",
        NACIMIENTO: "",
        CAN: "",
        DOMICILIO: "",
        MUNICIPIO: "",
        PROVINCIA: "",
        MUNICIPIO_N: "",
        PROVINCIA_N: "",
        PROGENITOR_1: "",
        PROGENITOR_2: "",
        EQUIPO: "",
        MRZ: "",
    }


def get_mrz() -> dict[str, str]:
    return {
        TIPO_DOC: "",
        PAIS: "",
        NUM_SOPORTE: "",
        HASH_1: "",
        NUM_DNI: "",
        OPT_DATA_1: "",
        NACIMIENTO: "",
        HASH_2: "",
        SEXO: "",
        VALIDEZ: "",
        HASH_3: "",
        NACIONALIDAD: "",
        OPT_DATA_2: "",
        FINAL_HASH: "",
        APELLIDOS: "",
        NOMBRE: ""
    }