from pop2vec.llm.src.new_code.utils import pad_after_x_abspos

import sys

if __name__ == '__main__':
    in_p = sys.argv[1]
    out_p = sys.argv[2]
    cutoff = int(sys.argv[3])
    pad_after_x_abspos(in_p, out_p, cutoff)