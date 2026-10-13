"""Check examples in docs/integer_arithmetic.md; run from the project root."""

from pathlib import Path
import torch
from model import MNISTMLP

cases = [(7,1,3,3,3),(-7,1,-4,-4,0),(65536,1,32768,32767,32767),(-65538,1,-32769,-32768,0),(32767,0,32767,32767,32767),(-32768,0,-32768,-32768,0)]
for acc, shift, expected_shift, expected_sat, expected_relu in cases:
    shifted = acc >> shift
    sat = min(32767, max(-32768, shifted))
    assert (shifted, sat, max(0,sat)) == (expected_shift,expected_sat,expected_relu)
assert [round(x) for x in (0.5,1.5,-0.5,-1.5)] == [0,2,0,-2]
assert (1*round(0.5*32) + 0*round(-0.25*32) + round(0.25*32)) >> 2 == 6
assert sum(bit << i for i,bit in enumerate([1,0,1,0,0,0,0,1])) == 0x85
for value,width,expected in [(-128,8,'80'),(-1,8,'ff'),(127,8,'7f'),(-1,32,'ffffffff'),(-32768,16,'8000'),(-1,16,'ffff'),(32767,16,'7fff')]:
    assert format(value & ((1<<width)-1), f'0{width//4}x') == expected
scores = [-8,-3,-3,-10,-7,-9,-6,-5,-4,-11]
best = 0
for i in range(1,10):
    if scores[i] > scores[best]:
        best = i
assert best == 1
checkpoint = torch.load(Path(__file__).resolve().parents[1] / 'artifacts' / 'mlp_float.pt', map_location='cpu', weights_only=True)
model = MNISTMLP()
model.load_state_dict(checkpoint['model_state_dict'])
assert model.fc1.weight.numel() == 50176 and model.fc2.weight.numel() == 640
assert model.fc1.bias.numel() == 64 and model.fc2.bias.numel() == 10
text = 'PASS: all six shift/saturation/ReLU examples, ties-to-even conversion, scale example, packed pixel order, seven signed hex encodings, signed argmax ties, and trained-model memory dimensions.\n'
Path('docs/step6_arithmetic_checks.txt').write_text(text, encoding='utf-8')
print(text)
