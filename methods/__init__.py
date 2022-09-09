from .TEXT.manager import TEXT
from .MISA.manager import MISA
from .MULT.manager import MULT
from .MAG_BERT.manager import MAG_BERT

method_map = {
    'text': TEXT,
    'misa': MISA,
    'mult': MULT,
    'mag_bert': MAG_BERT
}
