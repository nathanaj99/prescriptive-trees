\ Model FlowOPT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  0.6386475440940417 z[0,1] + 1.277308929974345 z[1,1]
   + 0.9166438358218501 z[2,1] + 1.380953304006865 z[3,1]
   + 0.4812705508895218 z[4,1] + 0.9808672502177167 z[5,1]
   + 0.6762337306140285 z[6,1] + 1.658281723509005 z[7,1]
   + 0.7752897906032368 z[8,1] + 1.07830919532638 z[9,1]
   + 1.662787061739968 z[10,1] + 1.59645542562363 z[11,1]
   + 0.61420669759637 z[12,1] + 1.323860018853412 z[13,1]
   + 1.478075802311722 z[14,1] + 0.606460085976385 z[15,1]
   + 2.27290961577137 z[16,1] + 1.606118110364785 z[17,1]
   + 0.1641264749893328 z[18,1] + 2.074194913971282 z[19,1]
   + 1.58321089565752 z[20,1] + 1.245510798608352 z[21,1]
   + 2.125855300070667 z[22,1] + 0.9140099774726 z[23,1]
   + 0.9526258879779633 z[24,1] + 1.132412350012665 z[25,1]
   + 1.312785953125963 z[26,1] + 0.5545367897563417 z[27,1]
   + 1.182267344071807 z[28,1] + 0.7253937740196201 z[29,1]
   + 1.2602772126999 z[30,1] + 0.9828946646107651 z[31,1]
   + 1.288042812035775 z[32,1] + 1.142446292902898 z[33,1]
   + 0.60377485834957 z[34,1] + 1.262156464389375 z[35,1]
   + 0.5732656116649224 z[36,1] + 0.5454529982578867 z[37,1]
   + 0.6746789436559499 z[38,1] + 1.097173908768243 z[39,1]
   + 0.9585336617854368 z[40,1] + 1.249551014576645 z[42,1]
   + 1.077126636198632 z[43,1] + 0.387088488986675 z[44,1]
   + 0.8819705623120985 z[45,1] + 1.71372790655287 z[46,1]
   + 0.76257417213559 z[47,1] + 0.7765228750239526 z[48,1]
   + 0.8299223165772618 z[49,1] + 0.2115173048324218 z[50,1]
   + 0.6483040275575225 z[51,1] + 0.9529068589199176 z[52,1]
   + 1.03383106674737 z[53,1] + 2.21567670237026 z[54,1]
   + 1.296716323121882 z[55,1] + 0.28729038125209 z[56,1]
   + 1.151412053784282 z[57,1] + 0.6973752738689951 z[58,1]
   + 1.007930058883718 z[59,1] + 1.279011604018765 z[60,1]
   + 0.8492927684039475 z[61,1] + 0.5073021378821067 z[62,1]
   + 1.01245678337806 z[63,1] + 0.4782265211037424 z[64,1]
   + 2.137726510630568 z[65,1] + 0.8463547574021284 z[66,1]
   + 2.177560179220225 z[67,1] + 0.5122140360228551 z[68,1]
   + 1.029486724243285 z[69,1] + 1.959003738432832 z[70,1]
   + 0.598480342091245 z[71,1] + 0.6465361460395118 z[72,1]
   + 0.6167147578667851 z[73,1] + 0.783150824933795 z[74,1]
   + 2.394196754290365 z[75,1] + 1.215349572476082 z[76,1]
   + 0.7226290101429924 z[77,1] + 0.8671185323951033 z[78,1]
   + 1.171597729103783 z[79,1] + 0.5525240833461924 z[80,1]
   + 1.522065079205303 z[81,1] + 1.350658436880163 z[82,1]
   + 0.7788855729770483 z[83,1] + 1.559709819383275 z[84,1]
   + 0.9011681902299568 z[85,1] + 0.39323664912784 z[86,1]
   + 1.34580822805736 z[87,1] + 1.185375899962403 z[88,1]
   + 0.4515754431564467 z[89,1] + 0.7584622084546001 z[90,1]
   + 0.91209675013666 z[91,1] + 1.265623356263433 z[92,1]
   + 1.30358591383535 z[93,1] + 1.520944645805552 z[94,1]
   + 0.9584727633672975 z[95,1] + 0.9887480231872735 z[96,1]
   + 1.588126667851035 z[97,1] + 1.21603532656797 z[98,1]
   + 1.856700938966073 z[99,1]
Subject To
 R0: p[1] = 1
 R1: - w[1,0] + zeta[0,1] <= 0
 R2: - w[1,1] + zeta[1,1] <= 0
 R3: - w[1,1] + zeta[2,1] <= 0
 R4: - w[1,0] + zeta[3,1] <= 0
 R5: - w[1,1] + zeta[4,1] <= 0
 R6: - w[1,0] + zeta[5,1] <= 0
 R7: - w[1,1] + zeta[6,1] <= 0
 R8: - w[1,0] + zeta[7,1] <= 0
 R9: - w[1,1] + zeta[8,1] <= 0
 R10: - w[1,1] + zeta[9,1] <= 0
 R11: - w[1,1] + zeta[10,1] <= 0
 R12: - w[1,0] + zeta[11,1] <= 0
 R13: - w[1,1] + zeta[12,1] <= 0
 R14: - w[1,0] + zeta[13,1] <= 0
 R15: - w[1,0] + zeta[14,1] <= 0
 R16: - w[1,0] + zeta[15,1] <= 0
 R17: - w[1,1] + zeta[16,1] <= 0
 R18: - w[1,0] + zeta[17,1] <= 0
 R19: - w[1,1] + zeta[18,1] <= 0
 R20: - w[1,0] + zeta[19,1] <= 0
 R21: - w[1,0] + zeta[20,1] <= 0
 R22: - w[1,1] + zeta[21,1] <= 0
 R23: - w[1,0] + zeta[22,1] <= 0
 R24: - w[1,1] + zeta[23,1] <= 0
 R25: - w[1,1] + zeta[24,1] <= 0
 R26: - w[1,0] + zeta[25,1] <= 0
 R27: - w[1,0] + zeta[26,1] <= 0
 R28: - w[1,0] + zeta[27,1] <= 0
 R29: - w[1,1] + zeta[28,1] <= 0
 R30: - w[1,1] + zeta[29,1] <= 0
 R31: - w[1,1] + zeta[30,1] <= 0
 R32: - w[1,1] + zeta[31,1] <= 0
 R33: - w[1,0] + zeta[32,1] <= 0
 R34: - w[1,0] + zeta[33,1] <= 0
 R35: - w[1,0] + zeta[34,1] <= 0
 R36: - w[1,1] + zeta[35,1] <= 0
 R37: - w[1,1] + zeta[36,1] <= 0
 R38: - w[1,0] + zeta[37,1] <= 0
 R39: - w[1,0] + zeta[38,1] <= 0
 R40: - w[1,1] + zeta[39,1] <= 0
 R41: - w[1,0] + zeta[40,1] <= 0
 R42: - w[1,0] + zeta[41,1] <= 0
 R43: - w[1,1] + zeta[42,1] <= 0
 R44: - w[1,0] + zeta[43,1] <= 0
 R45: - w[1,1] + zeta[44,1] <= 0
 R46: - w[1,0] + zeta[45,1] <= 0
 R47: - w[1,0] + zeta[46,1] <= 0
 R48: - w[1,1] + zeta[47,1] <= 0
 R49: - w[1,1] + zeta[48,1] <= 0
 R50: - w[1,1] + zeta[49,1] <= 0
 R51: - w[1,0] + zeta[50,1] <= 0
 R52: - w[1,1] + zeta[51,1] <= 0
 R53: - w[1,0] + zeta[52,1] <= 0
 R54: - w[1,0] + zeta[53,1] <= 0
 R55: - w[1,0] + zeta[54,1] <= 0
 R56: - w[1,1] + zeta[55,1] <= 0
 R57: - w[1,0] + zeta[56,1] <= 0
 R58: - w[1,0] + zeta[57,1] <= 0
 R59: - w[1,1] + zeta[58,1] <= 0
 R60: - w[1,0] + zeta[59,1] <= 0
 R61: - w[1,1] + zeta[60,1] <= 0
 R62: - w[1,1] + zeta[61,1] <= 0
 R63: - w[1,1] + zeta[62,1] <= 0
 R64: - w[1,1] + zeta[63,1] <= 0
 R65: - w[1,1] + zeta[64,1] <= 0
 R66: - w[1,0] + zeta[65,1] <= 0
 R67: - w[1,0] + zeta[66,1] <= 0
 R68: - w[1,0] + zeta[67,1] <= 0
 R69: - w[1,1] + zeta[68,1] <= 0
 R70: - w[1,0] + zeta[69,1] <= 0
 R71: - w[1,0] + zeta[70,1] <= 0
 R72: - w[1,0] + zeta[71,1] <= 0
 R73: - w[1,1] + zeta[72,1] <= 0
 R74: - w[1,0] + zeta[73,1] <= 0
 R75: - w[1,1] + zeta[74,1] <= 0
 R76: - w[1,0] + zeta[75,1] <= 0
 R77: - w[1,0] + zeta[76,1] <= 0
 R78: - w[1,0] + zeta[77,1] <= 0
 R79: - w[1,1] + zeta[78,1] <= 0
 R80: - w[1,0] + zeta[79,1] <= 0
 R81: - w[1,1] + zeta[80,1] <= 0
 R82: - w[1,1] + zeta[81,1] <= 0
 R83: - w[1,1] + zeta[82,1] <= 0
 R84: - w[1,0] + zeta[83,1] <= 0
 R85: - w[1,1] + zeta[84,1] <= 0
 R86: - w[1,1] + zeta[85,1] <= 0
 R87: - w[1,0] + zeta[86,1] <= 0
 R88: - w[1,1] + zeta[87,1] <= 0
 R89: - w[1,1] + zeta[88,1] <= 0
 R90: - w[1,1] + zeta[89,1] <= 0
 R91: - w[1,1] + zeta[90,1] <= 0
 R92: - w[1,1] + zeta[91,1] <= 0
 R93: - w[1,1] + zeta[92,1] <= 0
 R94: - w[1,0] + zeta[93,1] <= 0
 R95: - w[1,1] + zeta[94,1] <= 0
 R96: - w[1,0] + zeta[95,1] <= 0
 R97: - w[1,0] + zeta[96,1] <= 0
 R98: - w[1,1] + zeta[97,1] <= 0
 R99: - w[1,1] + zeta[98,1] <= 0
 R100: - w[1,1] + zeta[99,1] <= 0
 R101: - p[1] + w[1,0] + w[1,1] = 0
 R102: zeta[0,1] - z[0,1] = 0
 R103: zeta[1,1] - z[1,1] = 0
 R104: zeta[2,1] - z[2,1] = 0
 R105: zeta[3,1] - z[3,1] = 0
 R106: zeta[4,1] - z[4,1] = 0
 R107: zeta[5,1] - z[5,1] = 0
 R108: zeta[6,1] - z[6,1] = 0
 R109: zeta[7,1] - z[7,1] = 0
 R110: zeta[8,1] - z[8,1] = 0
 R111: zeta[9,1] - z[9,1] = 0
 R112: zeta[10,1] - z[10,1] = 0
 R113: zeta[11,1] - z[11,1] = 0
 R114: zeta[12,1] - z[12,1] = 0
 R115: zeta[13,1] - z[13,1] = 0
 R116: zeta[14,1] - z[14,1] = 0
 R117: zeta[15,1] - z[15,1] = 0
 R118: zeta[16,1] - z[16,1] = 0
 R119: zeta[17,1] - z[17,1] = 0
 R120: zeta[18,1] - z[18,1] = 0
 R121: zeta[19,1] - z[19,1] = 0
 R122: zeta[20,1] - z[20,1] = 0
 R123: zeta[21,1] - z[21,1] = 0
 R124: zeta[22,1] - z[22,1] = 0
 R125: zeta[23,1] - z[23,1] = 0
 R126: zeta[24,1] - z[24,1] = 0
 R127: zeta[25,1] - z[25,1] = 0
 R128: zeta[26,1] - z[26,1] = 0
 R129: zeta[27,1] - z[27,1] = 0
 R130: zeta[28,1] - z[28,1] = 0
 R131: zeta[29,1] - z[29,1] = 0
 R132: zeta[30,1] - z[30,1] = 0
 R133: zeta[31,1] - z[31,1] = 0
 R134: zeta[32,1] - z[32,1] = 0
 R135: zeta[33,1] - z[33,1] = 0
 R136: zeta[34,1] - z[34,1] = 0
 R137: zeta[35,1] - z[35,1] = 0
 R138: zeta[36,1] - z[36,1] = 0
 R139: zeta[37,1] - z[37,1] = 0
 R140: zeta[38,1] - z[38,1] = 0
 R141: zeta[39,1] - z[39,1] = 0
 R142: zeta[40,1] - z[40,1] = 0
 R143: zeta[41,1] - z[41,1] = 0
 R144: zeta[42,1] - z[42,1] = 0
 R145: zeta[43,1] - z[43,1] = 0
 R146: zeta[44,1] - z[44,1] = 0
 R147: zeta[45,1] - z[45,1] = 0
 R148: zeta[46,1] - z[46,1] = 0
 R149: zeta[47,1] - z[47,1] = 0
 R150: zeta[48,1] - z[48,1] = 0
 R151: zeta[49,1] - z[49,1] = 0
 R152: zeta[50,1] - z[50,1] = 0
 R153: zeta[51,1] - z[51,1] = 0
 R154: zeta[52,1] - z[52,1] = 0
 R155: zeta[53,1] - z[53,1] = 0
 R156: zeta[54,1] - z[54,1] = 0
 R157: zeta[55,1] - z[55,1] = 0
 R158: zeta[56,1] - z[56,1] = 0
 R159: zeta[57,1] - z[57,1] = 0
 R160: zeta[58,1] - z[58,1] = 0
 R161: zeta[59,1] - z[59,1] = 0
 R162: zeta[60,1] - z[60,1] = 0
 R163: zeta[61,1] - z[61,1] = 0
 R164: zeta[62,1] - z[62,1] = 0
 R165: zeta[63,1] - z[63,1] = 0
 R166: zeta[64,1] - z[64,1] = 0
 R167: zeta[65,1] - z[65,1] = 0
 R168: zeta[66,1] - z[66,1] = 0
 R169: zeta[67,1] - z[67,1] = 0
 R170: zeta[68,1] - z[68,1] = 0
 R171: zeta[69,1] - z[69,1] = 0
 R172: zeta[70,1] - z[70,1] = 0
 R173: zeta[71,1] - z[71,1] = 0
 R174: zeta[72,1] - z[72,1] = 0
 R175: zeta[73,1] - z[73,1] = 0
 R176: zeta[74,1] - z[74,1] = 0
 R177: zeta[75,1] - z[75,1] = 0
 R178: zeta[76,1] - z[76,1] = 0
 R179: zeta[77,1] - z[77,1] = 0
 R180: zeta[78,1] - z[78,1] = 0
 R181: zeta[79,1] - z[79,1] = 0
 R182: zeta[80,1] - z[80,1] = 0
 R183: zeta[81,1] - z[81,1] = 0
 R184: zeta[82,1] - z[82,1] = 0
 R185: zeta[83,1] - z[83,1] = 0
 R186: zeta[84,1] - z[84,1] = 0
 R187: zeta[85,1] - z[85,1] = 0
 R188: zeta[86,1] - z[86,1] = 0
 R189: zeta[87,1] - z[87,1] = 0
 R190: zeta[88,1] - z[88,1] = 0
 R191: zeta[89,1] - z[89,1] = 0
 R192: zeta[90,1] - z[90,1] = 0
 R193: zeta[91,1] - z[91,1] = 0
 R194: zeta[92,1] - z[92,1] = 0
 R195: zeta[93,1] - z[93,1] = 0
 R196: zeta[94,1] - z[94,1] = 0
 R197: zeta[95,1] - z[95,1] = 0
 R198: zeta[96,1] - z[96,1] = 0
 R199: zeta[97,1] - z[97,1] = 0
 R200: zeta[98,1] - z[98,1] = 0
 R201: zeta[99,1] - z[99,1] = 0
Bounds
Binaries
 p[1]
End
