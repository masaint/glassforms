// Written by tools/material_tables.py from the gathered optical data: do not edit by hand.
// Every table holds 31 values, at 400, 410, ... 700 nm.

use super::{Metal, Spectrum, SpectrumKind};

pub static METALS: &[Metal] = &[
    Metal {
        id: "silver", name: "silver",
        source: "Johnson & Christy 1972; refractiveindex.info database, database/data/main/Ag/nk/Johnson.yml",
        n: [0.05, 0.05, 0.0461, 0.04029, 0.04, 0.04, 0.04444, 0.04932, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05329, 0.0569, 0.05958, 0.0566, 0.05361, 0.05063, 0.05228, 0.05516, 0.05804, 0.05925, 0.05691, 0.05457, 0.05222, 0.04989, 0.04767, 0.04544, 0.04322, 0.041],
        k: [2.1035, 2.2325, 2.3478, 2.4566, 2.5528, 2.6484, 2.7511, 2.8545, 2.9476, 3.0391, 3.1309, 3.2233, 3.3157, 3.4101, 3.5047, 3.5974, 3.6786, 3.7598, 3.8409, 3.9249, 4.0097, 4.0944, 4.1768, 4.2543, 4.3318, 4.4094, 4.4869, 4.5658, 4.6447, 4.7236, 4.8025],
    },
    Metal {
        id: "aluminium", name: "aluminium",
        source: "Rakic 1995; refractiveindex.info database, database/data/main/Al/nk/Rakic.yml",
        n: [0.48787, 0.51308, 0.54105, 0.57037, 0.59969, 0.63324, 0.66843, 0.70362, 0.73927, 0.77592, 0.81257, 0.84921, 0.88783, 0.93029, 0.97274, 1.0152, 1.0576, 1.1062, 1.1583, 1.2103, 1.2623, 1.3143, 1.3664, 1.4303, 1.4942, 1.558, 1.627, 1.6975, 1.768, 1.8398, 1.9214],
        k: [4.8355, 4.96, 5.0843, 5.2085, 5.3328, 5.4543, 5.5748, 5.6953, 5.8146, 5.9313, 6.0481, 6.1648, 6.281, 6.3964, 6.5119, 6.6273, 6.7427, 6.8548, 6.965, 7.0753, 7.1855, 7.2957, 7.4059, 7.5081, 7.6102, 7.7124, 7.8042, 7.893, 7.9819, 8.0688, 8.142],
    },
    Metal {
        id: "gold", name: "gold",
        source: "Johnson & Christy 1972; refractiveindex.info database, database/data/main/Au/nk/Johnson.yml",
        n: [1.4684, 1.4621, 1.4561, 1.4503, 1.4174, 1.3831, 1.3489, 1.3148, 1.2152, 1.105, 0.97112, 0.80312, 0.63512, 0.55758, 0.48899, 0.42415, 0.38236, 0.34057, 0.29878, 0.27179, 0.24873, 0.22568, 0.20475, 0.18836, 0.17197, 0.15557, 0.13989, 0.13767, 0.13544, 0.13322, 0.131],
        k: [1.953, 1.9567, 1.9541, 1.9483, 1.9322, 1.9155, 1.8852, 1.8534, 1.8434, 1.8369, 1.8737, 1.9729, 2.0721, 2.2039, 2.3389, 2.472, 2.5938, 2.7156, 2.8374, 2.9561, 3.074, 3.1919, 3.3039, 3.4034, 3.5029, 3.6024, 3.7015, 3.7917, 3.882, 3.9722, 4.0624],
    },
    Metal {
        id: "copper", name: "copper",
        source: "Johnson & Christy 1972; refractiveindex.info database, database/data/main/Cu/nk/Johnson.yml",
        n: [1.3135, 1.2883, 1.2683, 1.2509, 1.2453, 1.2404, 1.2444, 1.2493, 1.2395, 1.2272, 1.2134, 1.1974, 1.1814, 1.1274, 1.0697, 1.0066, 0.9111, 0.81558, 0.72006, 0.60893, 0.49366, 0.37839, 0.294, 0.27527, 0.25653, 0.2378, 0.21989, 0.21767, 0.21544, 0.21322, 0.211],
        k: [2.1309, 2.1881, 2.2452, 2.3022, 2.3478, 2.3929, 2.4352, 2.4771, 2.5114, 2.5445, 2.5712, 2.5888, 2.6064, 2.5978, 2.5866, 2.5823, 2.6202, 2.6581, 2.696, 2.8181, 2.9624, 3.1068, 3.2456, 3.3725, 3.4995, 3.6264, 3.7521, 3.8539, 3.9556, 4.0574, 4.1592],
    },
    Metal {
        id: "chromium", name: "chrome",
        source: "Johnson & Christy 1974; refractiveindex.info database, database/data/main/Cr/nk/Johnson.yml",
        n: [2.015, 2.065, 2.1228, 2.1839, 2.253, 2.323, 2.411, 2.501, 2.5964, 2.6924, 2.7804, 2.8564, 2.9324, 3.0171, 3.1029, 3.1812, 3.1933, 3.2054, 3.2176, 3.2086, 3.1943, 3.18, 3.1643, 3.1452, 3.1262, 3.1071, 3.0891, 3.0802, 3.0713, 3.0624, 3.0536],
        k: [2.8487, 2.9112, 2.9728, 3.0339, 3.085, 3.135, 3.185, 3.235, 3.2616, 3.2856, 3.3048, 3.3168, 3.3288, 3.33, 3.33, 3.3291, 3.32, 3.3109, 3.3018, 3.3, 3.3, 3.3, 3.3029, 3.3124, 3.3219, 3.3314, 3.3411, 3.3522, 3.3633, 3.3744, 3.3856],
    },
    Metal {
        id: "nickel", name: "nickel",
        source: "Johnson & Christy 1974; refractiveindex.info database, database/data/main/Ni/nk/Johnson.yml",
        n: [1.7163, 1.7038, 1.7039, 1.7094, 1.719, 1.729, 1.7525, 1.7775, 1.7944, 1.8104, 1.8248, 1.8368, 1.8488, 1.8725, 1.8975, 1.9212, 1.9333, 1.9454, 1.9576, 1.9669, 1.9754, 1.984, 1.99, 1.99, 1.99, 1.99, 1.9916, 2.0071, 2.0227, 2.0382, 2.0538],
        k: [2.5925, 2.6675, 2.7406, 2.8128, 2.8785, 2.9435, 3.013, 3.083, 3.1476, 3.2116, 3.2772, 3.3452, 3.4132, 3.4811, 3.5489, 3.6158, 3.6733, 3.7309, 3.7885, 3.8503, 3.9131, 3.976, 4.0371, 4.0943, 4.1514, 4.2086, 4.2653, 4.3187, 4.372, 4.4253, 4.4787],
    },
    Metal {
        id: "platinum", name: "platinum",
        source: "Rakic et al. 1998 (Brendel-Bormann model); refractiveindex.info database, database/data/main/Pt/nk/Rakic-BB.yml",
        n: [1.7287, 1.753, 1.7766, 1.7998, 1.8227, 1.8453, 1.8679, 1.8906, 1.9135, 1.9366, 1.96, 1.9839, 2.0083, 2.0332, 2.0587, 2.0847, 2.1114, 2.1386, 2.1665, 2.1949, 2.2238, 2.2533, 2.2833, 2.3137, 2.3445, 2.3756, 2.4072, 2.439, 2.471, 2.5032, 2.5356],
        k: [2.8472, 2.9055, 2.9634, 3.0211, 3.0787, 3.1365, 3.1942, 3.252, 3.3099, 3.368, 3.4261, 3.4841, 3.5421, 3.6, 3.6578, 3.7153, 3.7726, 3.8294, 3.8859, 3.9419, 3.9974, 4.0523, 4.1065, 4.1601, 4.2131, 4.2655, 4.317, 4.3677, 4.4177, 4.4672, 4.5157],
    },
    Metal {
        id: "titanium", name: "titanium",
        source: "Johnson & Christy 1974; refractiveindex.info database, database/data/main/Ti/nk/Johnson.yml",
        n: [2.0913, 2.1288, 2.1672, 2.2061, 2.237, 2.267, 2.2925, 2.3175, 2.3344, 2.3504, 2.3728, 2.4048, 2.4368, 2.4721, 2.5079, 2.5418, 2.56, 2.5782, 2.5964, 2.616, 2.636, 2.656, 2.6764, 2.6979, 2.7193, 2.7407, 2.7622, 2.7844, 2.8067, 2.8289, 2.8511],
        k: [2.9556, 2.9744, 2.9917, 3.0083, 3.0235, 3.0385, 3.067, 3.097, 3.1324, 3.1684, 3.2076, 3.2516, 3.2956, 3.3418, 3.3882, 3.4346, 3.48, 3.5255, 3.5709, 3.612, 3.652, 3.692, 3.7286, 3.7571, 3.7857, 3.8143, 3.8427, 3.8693, 3.896, 3.9227, 3.9493],
    },
    Metal {
        id: "iron", name: "iron",
        source: "Johnson & Christy 1974; refractiveindex.info database, database/data/main/Fe/nk/Johnson.yml",
        n: [2.2606, 2.3294, 2.4006, 2.4728, 2.5295, 2.5845, 2.626, 2.666, 2.6952, 2.7232, 2.7592, 2.8072, 2.8552, 2.8889, 2.9211, 2.9497, 2.9467, 2.9436, 2.9406, 2.9263, 2.9091, 2.892, 2.8829, 2.8924, 2.9019, 2.9114, 2.9187, 2.9053, 2.892, 2.8787, 2.8653],
        k: [2.5931, 2.6369, 2.6733, 2.7067, 2.737, 2.767, 2.7925, 2.8175, 2.8416, 2.8656, 2.8848, 2.8968, 2.9088, 2.9164, 2.9236, 2.9318, 2.95, 2.9682, 2.9864, 3.0037, 3.0209, 3.038, 3.0536, 3.0655, 3.0774, 3.0893, 3.102, 3.122, 3.142, 3.162, 3.182],
    },
    Metal {
        id: "tungsten", name: "tungsten",
        source: "Weaver, Olson & Lynch 1975; refractiveindex.info database, database/data/main/W/nk/Weaver.yml",
        n: [3.4069, 3.3826, 3.3586, 3.3357, 3.3209, 3.3149, 3.3216, 3.3333, 3.3544, 3.3756, 3.4017, 3.4299, 3.4601, 3.4905, 3.5083, 3.5221, 3.523, 3.524, 3.5333, 3.5453, 3.5576, 3.5765, 3.5956, 3.617, 3.642, 3.6714, 3.7068, 3.7418, 3.7927, 3.835, 3.8683],
        k: [2.4159, 2.4159, 2.43, 2.4506, 2.4811, 2.516, 2.5547, 2.5929, 2.6251, 2.6572, 2.6796, 2.698, 2.7076, 2.7162, 2.728, 2.7414, 2.7637, 2.7859, 2.8096, 2.8352, 2.8607, 2.8789, 2.8958, 2.91, 2.9194, 2.9325, 2.9418, 2.9375, 2.9159, 2.896, 2.8803],
    },
];

pub static SPECTRA: &[Spectrum] = &[
    Spectrum {
        id: "ruby", name: "ruby", kind: SpectrumKind::Measured,
        source: "Mineral Spectroscopy Server, G. R. Rossman, California Institute of Technology (minerals.gps.caltech.edu), visible absorption data files: Ruby GRR 1843, Chanthaburi, Thailand, 0.787 mm thick, E perpendicular to c (file ru1843.a)",
        peak_per_cm: 45.524,
        shape: [0.9054, 1.0000, 0.9787, 0.8095, 0.5539, 0.3526, 0.2286, 0.1363, 0.1042, 0.1205, 0.1839, 0.2869, 0.4227, 0.5778, 0.7301, 0.8499, 0.9024, 0.8597, 0.7203, 0.5143, 0.3029, 0.1579, 0.0838, 0.0495, 0.0342, 0.0279, 0.0314, 0.0311, 0.0237, 0.0422, 0.0277],
    },
    Spectrum {
        id: "emerald", name: "emerald", kind: SpectrumKind::Measured,
        source: "Mineral Spectroscopy Server, G. R. Rossman, California Institute of Technology (minerals.gps.caltech.edu), visible absorption data files: Beryl var. emerald GRR 3570, Afghanistan, 0.859 mm thick; files Emerald_3570_a.asc (E perp c) and ...",
        peak_per_cm: 15.335,
        shape: [0.5637, 0.7287, 0.8923, 1.0000, 0.9940, 0.8944, 0.7238, 0.5298, 0.3695, 0.2276, 0.1547, 0.1347, 0.1470, 0.1821, 0.2411, 0.3272, 0.4380, 0.5660, 0.6972, 0.8140, 0.8873, 0.8887, 0.8138, 0.7054, 0.6529, 0.5156, 0.4113, 0.3295, 0.2876, 0.2065, 0.1431],
    },
    Spectrum {
        id: "sapphire", name: "blue sapphire", kind: SpectrumKind::Measured,
        source: "Mineral Spectroscopy Server, G. R. Rossman, California Institute of Technology (minerals.gps.caltech.edu), visible absorption data files: Sapphire GRR 1020, medium blue, locality unknown (probably Australia), 4.289 mm thick; files ...",
        peak_per_cm: 7.064,
        shape: [0.4287, 0.2877, 0.2714, 0.2957, 0.3926, 0.5500, 0.4992, 0.4087, 0.3593, 0.3732, 0.4243, 0.4874, 0.5548, 0.6200, 0.6798, 0.7310, 0.7712, 0.7994, 0.8164, 0.8246, 0.8270, 0.8276, 0.8296, 0.8342, 0.8431, 0.8570, 0.8757, 0.9001, 0.9287, 0.9619, 1.0000],
    },
    Spectrum {
        id: "amethyst", name: "amethyst", kind: SpectrumKind::Measured,
        source: "Mineral Spectroscopy Server, G. R. Rossman, California Institute of Technology (minerals.gps.caltech.edu), visible absorption data files: Amethyst region of an ametrine, Anahi Mine, Bolivia, 6.17 mm slab cut on (0001); file ameth.bo",
        peak_per_cm: 0.917,
        shape: [1.0000, 0.8561, 0.6903, 0.5278, 0.4198, 0.3860, 0.4057, 0.4613, 0.5387, 0.6260, 0.7088, 0.7863, 0.8430, 0.8800, 0.8899, 0.8822, 0.8462, 0.7972, 0.7285, 0.6576, 0.5823, 0.5071, 0.4417, 0.3773, 0.3173, 0.2595, 0.2061, 0.1668, 0.1320, 0.0927, 0.0742],
    },
    Spectrum {
        id: "citrine", name: "citrine", kind: SpectrumKind::Digitised,
        source: "Mineral Spectroscopy Server, G. R. Rossman, Caltech: Citrine (heated amethyst) GRR 3910, Rio Grande do Sul, Brazil, 2.29 mm thick. Only a plot is offered (Citrine_3910_spectrum.jpg), no data file",
        peak_per_cm: 28.26,
        shape: [1.0000, 0.9020, 0.8139, 0.7343, 0.6362, 0.5796, 0.5393, 0.5032, 0.4632, 0.4133, 0.3517, 0.3011, 0.2640, 0.2346, 0.2088, 0.1854, 0.1614, 0.1387, 0.1192, 0.1023, 0.0895, 0.0782, 0.0686, 0.0598, 0.0531, 0.0474, 0.0418, 0.0364, 0.0318, 0.0283, 0.0248],
    },
    Spectrum {
        id: "smoky", name: "smoky quartz", kind: SpectrumKind::Measured,
        source: "Mineral Spectroscopy Server, G. R. Rossman, California Institute of Technology (minerals.gps.caltech.edu), visible absorption data files: Smoky quartz GRR 2918, Jack Rabbit Claim (Crystal Peak), Teller County, Colorado, 5.788 mm thick; ...",
        peak_per_cm: 3.895,
        shape: [1.0000, 0.9920, 0.9846, 0.9802, 0.9772, 0.9748, 0.9720, 0.9666, 0.9589, 0.9463, 0.9302, 0.9104, 0.8875, 0.8626, 0.8370, 0.8121, 0.7877, 0.7648, 0.7433, 0.7225, 0.7017, 0.6814, 0.6606, 0.6393, 0.6175, 0.5949, 0.5707, 0.5463, 0.5217, 0.4970, 0.4716],
    },
    Spectrum {
        id: "topaz", name: "brown topaz", kind: SpectrumKind::Measured,
        source: "Mineral Spectroscopy Server, G. R. Rossman, California Institute of Technology (minerals.gps.caltech.edu), visible absorption data files: Topaz GRR 515, natural brown ('sherry'), Thomas Mountains (Topaz Mountain), Utah, plotted as 7.00 ...",
        peak_per_cm: 1.954,
        shape: [1.0000, 0.9898, 0.9862, 0.9826, 0.9806, 0.9749, 0.9601, 0.9360, 0.9089, 0.8777, 0.8337, 0.7871, 0.7400, 0.6924, 0.6448, 0.5983, 0.5563, 0.5036, 0.4585, 0.4130, 0.3772, 0.3408, 0.3071, 0.2743, 0.2431, 0.2190, 0.1965, 0.1796, 0.1679, 0.1561, 0.1474],
    },
    Spectrum {
        id: "cobalt", name: "cobalt blue glass", kind: SpectrumKind::Measured,
        source: "SCHOTT Advanced Optics online shop, glass page 'BG3': reference internal transmittance at 1.0 mm, tabulated every 10 nm (the page's 'Internal Transmittance' chart data)",
        peak_per_cm: 121.85,
        shape: [0.0040, 0.0071, 0.0109, 0.0164, 0.0234, 0.0332, 0.0473, 0.0747, 0.1221, 0.1933, 0.2640, 0.3629, 0.4774, 0.6171, 0.6842, 0.6254, 0.5392, 0.5812, 0.7666, 0.9899, 0.9914, 0.9438, 0.9281, 0.9554, 1.0000, 0.9700, 0.8356, 0.6204, 0.3816, 0.1926, 0.0841],
    },
    Spectrum {
        id: "didymium", name: "didymium glass", kind: SpectrumKind::Measured,
        source: "SCHOTT Advanced Optics online shop, glass page 'BG36': reference internal transmittance at 1.0 mm, tabulated every 10 nm (the page's 'Internal Transmittance' chart data)",
        peak_per_cm: 110.37,
        shape: [0.0067, 0.0047, 0.0085, 0.0622, 0.1179, 0.1104, 0.0936, 0.1230, 0.0948, 0.0316, 0.0450, 0.1614, 0.1647, 0.1791, 0.0312, 0.0091, 0.0162, 0.3604, 1.0000, 0.5718, 0.1288, 0.0146, 0.0092, 0.0106, 0.0032, 0.0003, 0.0014, 0.0184, 0.0495, 0.0235, 0.0067],
    },
    Spectrum {
        id: "aqua", name: "copper aqua glass", kind: SpectrumKind::Measured,
        source: "SCHOTT Advanced Optics online shop, glass page 'BG39': reference internal transmittance at 1.0 mm, tabulated every 10 nm (the page's 'Internal Transmittance' chart data)",
        peak_per_cm: 46.517,
        shape: [0.0310, 0.0263, 0.0224, 0.0188, 0.0160, 0.0137, 0.0116, 0.0098, 0.0082, 0.0073, 0.0065, 0.0065, 0.0073, 0.0095, 0.0132, 0.0195, 0.0292, 0.0431, 0.0622, 0.0876, 0.1202, 0.1612, 0.2113, 0.2711, 0.3420, 0.4238, 0.5181, 0.6221, 0.7361, 0.8606, 1.0000],
    },
    Spectrum {
        id: "signal", name: "selenium ruby glass", kind: SpectrumKind::Measured,
        source: "SCHOTT Advanced Optics online shop, glass page 'RG610': reference internal transmittance at 3.0 mm, tabulated every 10 nm (the page's 'Internal Transmittance' chart data)",
        peak_per_cm: 38.376,
        shape: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.9348, 0.5665, 0.3014, 0.1390, 0.0574, 0.0227, 0.0096, 0.0049, 0.0030, 0.0022, 0.0019, 0.0017, 0.0016, 0.0016],
    },
    Spectrum {
        id: "yellow", name: "cadmium yellow glass", kind: SpectrumKind::Measured,
        source: "SCHOTT Advanced Optics online shop, glass page 'GG495': reference internal transmittance at 3.0 mm, tabulated every 10 nm (the page's 'Internal Transmittance' chart data)",
        peak_per_cm: 38.376,
        shape: [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5117, 0.1322, 0.0271, 0.0074, 0.0034, 0.0022, 0.0016, 0.0013, 0.0010, 0.0010, 0.0009, 0.0010, 0.0010, 0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0016, 0.0017, 0.0018, 0.0019, 0.0021],
    },
    Spectrum {
        id: "amber", name: "amber bottle glass", kind: SpectrumKind::Digitised,
        source: "M. Mueller, C. Ruessel, O. Claussen, UV-VIS spectroscopic investigations of amber glass at high temperatures, Glastech. Ber. Glass Sci. Technol. 72 (1999) No. 11, 362-366, Figure 1, curve 5 (amber glass, room temperature). Open access ...",
        peak_per_cm: 11.89,
        shape: [0.9907, 1.0000, 0.9840, 0.9479, 0.8999, 0.8377, 0.7696, 0.6972, 0.6291, 0.5694, 0.5122, 0.4584, 0.4180, 0.3818, 0.3516, 0.3280, 0.3120, 0.2994, 0.2876, 0.2809, 0.2792, 0.2775, 0.2750, 0.2801, 0.2792, 0.2792, 0.2809, 0.2826, 0.2843, 0.2860, 0.2876],
    },
    Spectrum {
        id: "manganese", name: "manganese purple glass", kind: SpectrumKind::Digitised,
        source: "A.-I. Bidegaray, S. Godet, M. Bogaerts, P. Cosyns, K. Nys, H. Terryn, A. Ceglia, To be purple or not to be purple? How different production parameters influence colour and redox in manganese containing glass, J. Archaeol. Sci. Rep. 27 ...",
        peak_per_cm: 14.11,
        shape: [0.2126, 0.2884, 0.4096, 0.5344, 0.6697, 0.8051, 0.8561, 0.9291, 0.9546, 1.0000, 0.9830, 0.9639, 0.9454, 0.8880, 0.8412, 0.7902, 0.7314, 0.6733, 0.6230, 0.5734, 0.5223, 0.4805, 0.4486, 0.4217, 0.3976, 0.3770, 0.3579, 0.3409, 0.3189, 0.3005, 0.2807],
    },
    Spectrum {
        id: "chromegreen", name: "chromium green glass", kind: SpectrumKind::Digitised,
        source: "Figure 'Fig. 2.5. Linear absorption coefficients of chromium oxide in soda-lime-silica glass' (Cr3+ curve), reproduced in J. E. Shelby, 'Lecture 4. Color in Commercial Glasses', Glass Processing Course, International Materials Institute ...",
        peak_per_cm: 0.0,
        shape: [0.3132, 0.4005, 0.5841, 0.7670, 0.9484, 1.0000, 0.9389, 0.8448, 0.7473, 0.6490, 0.5491, 0.4460, 0.3499, 0.2614, 0.1883, 0.1547, 0.2002, 0.2760, 0.3717, 0.5149, 0.6998, 0.8193, 0.8913, 0.9201, 0.9621, 0.9439, 0.8935, 0.8801, 0.8319, 0.7440, 0.6250],
    },
    Spectrum {
        id: "cranberry", name: "gold ruby (cranberry) glass", kind: SpectrumKind::Approximate,
        source: "COMPUTED, not measured: quasi-static (small-particle) limit of Mie theory for gold nanospheres in glass of index 1.50, using the gold optical constants of P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972) from this file's ...",
        peak_per_cm: 0.0,
        shape: [0.4559, 0.4475, 0.4387, 0.4300, 0.4301, 0.4309, 0.4298, 0.4281, 0.4550, 0.4909, 0.5585, 0.6969, 0.9031, 1.0000, 0.9570, 0.7706, 0.5581, 0.3832, 0.2563, 0.1780, 0.1266, 0.0909, 0.0670, 0.0517, 0.0400, 0.0309, 0.0239, 0.0207, 0.0179, 0.0156, 0.0137],
    },
    Spectrum {
        id: "uranium", name: "uranium glass", kind: SpectrumKind::Approximate,
        source: "MY APPROXIMATION from text descriptions, not data. Band position from the uranyl ion's charge-transfer absorption 'near 420 nm, at the blue edge of the visible' (Wikipedia, 'Uranyl', which cites the uranyl spectroscopy literature); ...",
        peak_per_cm: 0.0,
        shape: [1.0000, 0.8344, 0.7210, 0.5957, 0.4470, 0.2970, 0.1728, 0.0881, 0.0396, 0.0160, 0.0061, 0.0024, 0.0010, 0.0005, 0.0003, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    },
    Spectrum {
        id: "bronze", name: "bronze glass", kind: SpectrumKind::Measured,
        source: "M. Rubin. Optical properties of soda lime silica glasses. Sol. Energy Mater. 12, 275-288 (1985) (Bronze soda lime silica window glass), as tabulated in the refractiveindex.info database",
        peak_per_cm: 1.2058,
        shape: [0.8015, 0.8464, 0.8898, 0.9396, 0.9728, 0.9796, 0.9864, 0.9969, 1.0000, 0.9990, 0.9844, 0.9575, 0.9157, 0.8532, 0.7903, 0.7058, 0.6633, 0.6856, 0.6971, 0.6927, 0.6866, 0.6713, 0.6687, 0.6897, 0.7025, 0.7087, 0.7029, 0.6772, 0.6553, 0.6334, 0.6232],
    },
    Spectrum {
        id: "grey", name: "grey glass", kind: SpectrumKind::Measured,
        source: "M. Rubin. Optical properties of soda lime silica glasses. Sol. Energy Mater. 12, 275-288 (1985) (Grey soda lime silica window glass), as tabulated in the refractiveindex.info database",
        peak_per_cm: 1.4157,
        shape: [0.5326, 0.6198, 0.7006, 0.7830, 0.8392, 0.8531, 0.8652, 0.8701, 0.8842, 0.9150, 0.9398, 0.9691, 0.9704, 0.9359, 0.8899, 0.7929, 0.7675, 0.8594, 0.9276, 0.9650, 0.9840, 0.9683, 0.9657, 0.9965, 1.0000, 0.9817, 0.9257, 0.8145, 0.7127, 0.6072, 0.5339],
    },
    Spectrum {
        id: "water", name: "water", kind: SpectrumKind::Measured,
        source: "R. M. Pope and E. S. Fry, Absorption spectrum (380-700 nm) of pure water. II. Integrating cavity measurements, Appl. Opt. 36, 8710-8723 (1997); table as distributed by the Oregon Medical Laser Center (omlc.org)",
        peak_per_cm: 0.00624,
        shape: [0.0106, 0.0076, 0.0073, 0.0079, 0.0102, 0.0148, 0.0157, 0.0170, 0.0204, 0.0240, 0.0327, 0.0521, 0.0655, 0.0696, 0.0760, 0.0905, 0.0992, 0.1114, 0.1436, 0.2165, 0.3564, 0.4237, 0.4415, 0.4673, 0.4981, 0.5449, 0.6571, 0.7035, 0.7452, 0.8269, 1.0000],
    },
];
