// Unpolarized light is an equal mixture of independent s and p powers. The
// coherent film sits on the coat, or directly on the pigment when there is no coat.
// q contains the s and p optical admittances. All indices are real and lossless.
fn paintFilmPolar(c: f32, incident: f32, film: f32, substrate: f32, nm: f32, wavelength: f32) -> vec2f {
  if (incident==film && film==substrate) { return vec2f(0.0); }
  let tangent2 = incident*incident*max(0.0,1.0-c*c);
  let transmitted = substrate*substrate-tangent2;
  if (transmitted <= 0.0 || c <= 0.0) { return vec2f(1.0); }
  let q0 = c * vec2f(incident,1.0/incident);
  let q3 = sqrt(transmitted) * vec2f(1.0,1.0/(substrate*substrate));
  let q03 = q0*q3;
  let summed = q0+q3;
  let normal2 = film*film-tangent2;
  let scale = vec2f(1.0,1.0/(film*film));
  let distance = 6.28318530718*nm/wavelength;
  var power: vec2f;
  if (abs(normal2) < 1e-7) {
    // At the film's critical angle sin(delta)/q tends to distance/scale.
    let limit = distance*q03/scale;
    power = 4.0*q03/(summed*summed+limit*limit);
  } else if (normal2 > 0.0) {
    let normal = sqrt(normal2);
    let q2 = normal*scale;
    let delta = distance*normal;
    let real = summed*cos(delta);
    let imaginary = (q2+q03/q2)*sin(delta);
    power = 4.0*q03/(real*real+imaginary*imaginary);
  } else {
    // A finite evanescent film still couples light to a propagating substrate.
    // Divide the transfer matrix by cosh(delta) to avoid overflow in thick films.
    let decay = sqrt(-normal2);
    let q2 = decay*scale;
    let attenuation = exp(-2.0*distance*decay);
    let reciprocal = 1.0/(1.0+attenuation);
    let imaginary = (q03/q2-q2)*(1.0-attenuation)*reciprocal;
    let sech2 = 4.0*attenuation*reciprocal*reciprocal;
    power = 4.0*q03*sech2/(summed*summed+imaginary*imaginary);
  }
  return clamp(vec2f(1.0)-power,vec2f(0.0),vec2f(1.0));
}

// Per wavelength, order the shares as film reflection, return from the substrate
// through the coat, then transmission to the pigmented body.
fn paintFilmWeights(c: f32, incident: f32, body: f32, coat: f32, coated: bool,
                    film: f32, nm: f32, wavelength: f32) -> vec3f {
  if (nm <= 0.0) { return paintLayerWeights(c,incident,body,coat,coated); }
  let outer = select(body,coat,coated);
  let top = paintFilmPolar(c,incident,film,outer,nm,wavelength);
  if (!coated || coat==body) {
    let reflection = 0.5*(top.x+top.y);
    return vec3f(reflection,0.0,1.0-reflection);
  }
  let cc = paintRefractedCosine(c,incident,coat);
  let cb = paintRefractedCosine(c,incident,body);
  return paintLayerReturns(top,paintPolarPower(cc,cb,coat,body));
}

struct PaintFilmLanes { top: vec4f, substrate: vec4f, diffuse: vec4f, }
fn paintFilmLanes(c: f32, incident: f32, body: f32, coat: f32, coated: bool,
                  film: f32, nm: f32, wavelengths: vec4f) -> PaintFilmLanes {
  let x=paintFilmWeights(c,incident,body,coat,coated,film,nm,wavelengths.x);
  let y=paintFilmWeights(c,incident,body,coat,coated,film,nm,wavelengths.y);
  let z=paintFilmWeights(c,incident,body,coat,coated,film,nm,wavelengths.z);
  let w=paintFilmWeights(c,incident,body,coat,coated,film,nm,wavelengths.w);
  return PaintFilmLanes(vec4f(x.x,y.x,z.x,w.x),vec4f(x.y,y.y,z.y,w.y),vec4f(x.z,y.z,z.z,w.z));
}

// CIE observer at the 24 wavelength-bin centres from 400 to 700 nm.
const paintFilmObserver = array<vec3f,24>(
  vec3f(0.0303610392,-0.0199703467,0.136779476),
  vec3f(0.114889036,-0.089229774,0.614387789),
  vec3f(0.187933458,-0.198028753,1.56373596),
  vec3f(0.248885848,-0.23121956,1.88501168),
  vec3f(0.066757264,-0.140560923,1.8382955),
  vec3f(-0.146265087,0.0183071671,1.42604521),
  vec3f(-0.309548025,0.219447937,0.774626413),
  vec3f(-0.522801088,0.469838183,0.350548506),
  vec3f(-0.735108432,0.807033867,0.116457491),
  vec3f(-0.895947353,1.2235894,-0.0383966855),
  vec3f(-0.814382823,1.48868382,-0.128772809),
  vec3f(-0.415550107,1.50287706,-0.164345594),
  vec3f(0.208981547,1.3491398,-0.168154611),
  vec3f(0.966135493,1.06639878,-0.15176735),
  vec3f(1.71136029,0.70610409,-0.122871185),
  vec3f(2.27776428,0.332829047,-0.0878811372),
  vec3f(2.50261436,0.0310771346,-0.0546855046),
  vec3f(2.23987061,-0.119120259,-0.0305340979),
  vec3f(1.65628706,-0.139189194,-0.0161726486),
  vec3f(1.02132683,-0.0952762313,-0.00878834096),
  vec3f(0.526619393,-0.0445298571,-0.00510789238),
  vec3f(0.226641066,-0.012534181,-0.00302964429),
  vec3f(0.0808211712,0.000400800586,-0.00169110706),
  vec3f(0.0234652503,0.00291879127,-0.000842387401)
);

// The CIE observer integrated across 96 evenly spaced inverse-wavelength bands.
const paintFilmBandWeight = array<vec3f,96>(
  vec3f(0.0006713548,0.000150469148,-3.7963542e-05),
  vec3f(0.00119137779,0.000151550458,-5.21248564e-05),
  vec3f(0.00200760428,0.000116498793,-6.95006987e-05),
  vec3f(0.00323024176,2.36801628e-05,-9.02058007e-05),
  vec3f(0.00498295585,-0.000151626046,-0.000114310184),
  vec3f(0.00739276213,-0.000434255977,-0.000141958921),
  vec3f(0.0105756087,-0.000844861603,-0.000173551125),
  vec3f(0.0146188098,-0.00139467046,-0.000209958241),
  vec3f(0.0195624254,-0.00208000796,-0.000252745231),
  vec3f(0.0253823605,-0.00287751126,-0.000304346819),
  vec3f(0.0319781844,-0.00374102779,-0.000368148506),
  vec3f(0.0391683139,-0.0046010536,-0.000448430734),
  vec3f(0.0466942963,-0.00536723533,-0.00055015377),
  vec3f(0.0542346032,-0.00593398931,-0.000678587256),
  vec3f(0.0614268647,-0.00618877599,-0.000838816494),
  vec3f(0.0678961288,-0.00602211042,-0.00103518149),
  vec3f(0.0732857755,-0.00533808392,-0.00127071949),
  vec3f(0.0772873302,-0.00406407408,-0.0015466841),
  vec3f(0.0796656505,-0.00215844313,-0.00186220406),
  vec3f(0.0802767227,0.000384666996,-0.00221412446),
  vec3f(0.0790764358,0.00353390688,-0.00259704709),
  vec3f(0.0761239278,0.00722116729,-0.00300346941),
  vec3f(0.0719108891,0.0112188456,-0.00341644442),
  vec3f(0.0668217845,0.0153444621,-0.00382202611),
  vec3f(0.0610223098,0.019503443,-0.00421188881),
  vec3f(0.0546890766,0.0236028916,-0.00457810805),
  vec3f(0.0480011132,0.0275548709,-0.00491333304),
  vec3f(0.0411322886,0.0312791092,-0.0052109033),
  vec3f(0.034244973,0.0347050892,-0.00546491319),
  vec3f(0.0274851105,0.0377735279,-0.00567023126),
  vec3f(0.0209453234,0.0404887434,-0.00582837491),
  vec3f(0.0146974301,0.0428658796,-0.00594137631),
  vec3f(0.00883164018,0.0448694085,-0.0060046808),
  vec3f(0.00341740202,0.0464735472,-0.00601403618),
  vec3f(-0.00149593133,0.0476624861,-0.00596552004),
  vec3f(-0.00587768375,0.0484302616,-0.00585555481),
  vec3f(-0.00971390338,0.0487803238,-0.00568091527),
  vec3f(-0.0130050636,0.0487248274,-0.0054387295),
  vec3f(-0.0157635897,0.0482836795,-0.00512647581),
  vec3f(-0.0180113622,0.0474833949,-0.00474197862),
  vec3f(-0.0197773187,0.0463558146,-0.00428340629),
  vec3f(-0.021019932,0.0448208407,-0.00373599939),
  vec3f(-0.0215993162,0.0426451269,-0.00306749658),
  vec3f(-0.0216035008,0.0399453725,-0.00228590079),
  vec3f(-0.0211504618,0.0368844492,-0.00140476624),
  vec3f(-0.0203618972,0.0336314388,-0.000438596815),
  vec3f(-0.0193498122,0.0303410193,0.000599722787),
  vec3f(-0.0182072682,0.0271389135,0.00170107409),
  vec3f(-0.0170041157,0.0241146862,0.00286176163),
  vec3f(-0.0157873451,0.0213213661,0.0040847611),
  vec3f(-0.014584866,0.0187801651,0.00538065709),
  vec3f(-0.0134031567,0.0164850629,0.00676841901),
  vec3f(-0.012227295,0.0144089722,0.00827551855),
  vec3f(-0.0110656319,0.0125263297,0.00993589464),
  vec3f(-0.00993086363,0.0108107042,0.011787616),
  vec3f(-0.00883874704,0.00923788173,0.0138679281),
  vec3f(-0.0078064142,0.00778771675,0.0162061013),
  vec3f(-0.00684923859,0.00644455405,0.0188148087),
  vec3f(-0.00597712049,0.00519672795,0.0216814709),
  vec3f(-0.00519119399,0.00403561926,0.0247615378),
  vec3f(-0.00448189576,0.00295465485,0.0279757508),
  vec3f(-0.00382905999,0.00194850747,0.0312128968),
  vec3f(-0.00320425228,0.00101262018,0.0343385027),
  vec3f(-0.00257502722,0.000143062417,0.0372085624),
  vec3f(-0.00191032276,-0.000663363271,0.0396861361),
  vec3f(-0.00118591817,-0.00140889412,0.0416578739),
  vec3f(-0.000388859156,-0.00209459854,0.0430474403),
  vec3f(0.00047450917,-0.00271972106,0.0438389444),
  vec3f(0.00130175912,-0.00327376016,0.044293753),
  vec3f(0.00205093558,-0.00375299994,0.0445103378),
  vec3f(0.00271439268,-0.00415783067,0.0444982831),
  vec3f(0.00328613632,-0.0044893598,0.0442703813),
  vec3f(0.00376203482,-0.00474944727,0.0438419957),
  vec3f(0.00413995427,-0.00494070875,0.0432304156),
  vec3f(0.00441981028,-0.00506648781,0.0424542305),
  vec3f(0.00453218864,-0.00510389792,0.0415311189),
  vec3f(0.00433830981,-0.00500377052,0.0404773266),
  vec3f(0.00390223458,-0.00478478597,0.0392250547),
  vec3f(0.00347950136,-0.00449030411,0.0372165484),
  vec3f(0.00312699581,-0.00414456463,0.0344724446),
  vec3f(0.00284218695,-0.00376596881,0.0311858337),
  vec3f(0.00261081151,-0.0033708889,0.0275754655),
  vec3f(0.00241316249,-0.00297336529,0.0238579162),
  vec3f(0.00222973156,-0.00258502199,0.020224563),
  vec3f(0.00204524349,-0.00221511174,0.0168262038),
  vec3f(0.00185070939,-0.00187062928,0.0137661392),
  vec3f(0.00164364543,-0.00155646199,0.0111008222),
  vec3f(0.00142694553,-0.00127556929,0.00884607637),
  vec3f(0.0012070294,-0.00102919308,0.00698647641),
  vec3f(0.000991843959,-0.000817101407,0.00548565225),
  vec3f(0.000789142234,-0.000637861085,0.00429581718),
  vec3f(0.000605271259,-0.000489127762,0.00336549074),
  vec3f(0.000444525929,-0.000367936115,0.00264501521),
  vec3f(0.000309000408,-0.00027097168,0.00208993909),
  vec3f(0.000198801867,-0.000194808218,0.00166262786),
  vec3f(0.000112474912,-0.000136099661,0.00133258078)
);

// Integrate 1/(a+b*cos(phase)) over an interval in inverse wavelength.
// Its primitive advances one full cycle each optical period. Using atan2
// keeps the partial period continuous through bright, narrow resonances.
fn paintFilmCycle(cycles: f32, factor: vec2f) -> vec2f {
  let halfAngle = 3.14159265359 * fract(cycles);
  return floor(cycles) + atan2(factor*sin(halfAngle),vec2f(cos(halfAngle))) / 3.14159265359;
}

fn paintFilmBandRGB(c: f32, incident: f32, body: f32, coat: f32, coated: bool,
                    opticalDepth: f32, products: vec2f, topPass: vec2f) -> mat3x3f {
  let outer = select(body,coat,coated);
  let cosineOuter = paintRefractedCosine(c,incident,outer);
  let cosineBody = paintRefractedCosine(c,incident,body);
  let base = select(vec2f(0.0),paintPolarPower(cosineOuter,cosineBody,outer,body),coated);
  let b = vec2f(1.0) + products*products;
  let f = 2.0*products;
  let meanTopPass = topPass / max(vec2f(1e-10),vec2f(1.0)-products*products);
  let topFactor = abs((vec2f(1.0)-products)/(vec2f(1.0)+products));
  let db = (1.0-base)*b + base*topPass;
  let df = (1.0-base)*f;
  let diffuseFactor = sqrt(max(vec2f(0.0),(db-df)/max(db+df,vec2f(1e-15))));
  let meanDiffuse = topPass*(1.0-base)/sqrt(max(vec2f(1e-20),(db-df)*(db+df)));
  let start = 2.0*opticalDepth/700.0;
  let step = 2.0*opticalDepth*(1.0/400.0-1.0/700.0)/96.0;
  var previousTop = paintFilmCycle(start,topFactor);
  var previousDiffuse = paintFilmCycle(start,diffuseFactor);
  var top = vec3f(0.0);
  var diffuse = vec3f(0.0);
  for (var j=0u; j<96u; j++) {
    let end = start + f32(j+1u)*step;
    let nextTop = paintFilmCycle(end,topFactor);
    let nextDiffuse = paintFilmCycle(end,diffuseFactor);
    let t = meanTopPass*(nextTop-previousTop)/step;
    let d = meanDiffuse*(nextDiffuse-previousDiffuse)/step;
    top += paintFilmBandWeight[j]*(1.0-0.5*(t.x+t.y));
    diffuse += paintFilmBandWeight[j]*0.5*(d.x+d.y);
    previousTop = nextTop;
    previousDiffuse = nextDiffuse;
  }
  top = clamp(top,vec3f(0.0),vec3f(1.0));
  if (!coated || coat==body) { return mat3x3f(top,vec3f(0.0),vec3f(1.0)-top); }
  let under = clamp(vec3f(1.0)-top-diffuse,vec3f(0.0),vec3f(1.0)-top);
  return mat3x3f(top,under,vec3f(1.0)-top-under);
}

// The RGB passes integrate neutral illumination against the same CIE observer
// as single-body spectral transport. Its three columns sum to unit white.
fn paintFilmRGB(c: f32, incident: f32, body: f32, coat: f32, coated: bool,
                film: f32, nm: f32) -> mat3x3f {
  if (nm<=0.0) {
    let shares=paintLayerWeights(c,incident,body,coat,coated);
    return mat3x3f(vec3f(shares.x),vec3f(shares.y),vec3f(shares.z));
  }
  let outer = select(body,coat,coated);
  let tangent2 = incident*incident*max(0.0,1.0-c*c);
  if (tangent2 >= outer*outer && !(incident==outer && c==0.0)) {
    return mat3x3f(vec3f(1.0),vec3f(0.0),vec3f(0.0));
  }
  if (film*film>tangent2 && c>0.0) {
    let normal = sqrt(film*film-tangent2);
    let opticalDepth = nm*normal;
    let q0 = c*vec2f(incident,1.0/incident);
    let qf = normal*vec2f(1.0,1.0/(film*film));
    let qo = sqrt(max(0.0,outer*outer-tangent2))*vec2f(1.0,1.0/(outer*outer));
    let r01 = (q0-qf)/(q0+qf);
    let r12 = (qf-qo)/(qf+qo);
    let products = r01*r12;
    let resonant = max(abs(products.x),abs(products.y));
    // A thick or strongly resonant coating can change colour several times
    // across one RGB wavelength bin. Average its repeated paths inside each bin.
    if (opticalDepth*(1.0+3.0*resonant) > 2300.0*(1.0-resonant)) {
      return paintFilmBandRGB(c,incident,body,coat,coated,opticalDepth,products,(1.0-r01*r01)*(1.0-r12*r12));
    }
  }
  var top=vec3f(0.0);
  var base=vec3f(0.0);
  var white=vec3f(0.0);
  for(var j=0u;j<24u;j++) {
    let wavelength=400.0+(f32(j)+0.5)*12.5;
    let weight=paintFilmObserver[j];
    let power=paintFilmWeights(c,incident,body,coat,coated,film,nm,wavelength);
    top+=weight*power.x; base+=weight*power.y; white+=weight;
  }
  top=clamp(top/white,vec3f(0.0),vec3f(1.0));
  base=clamp(base/white,vec3f(0.0),vec3f(1.0)-top);
  return mat3x3f(top,base,vec3f(1.0)-top-base);
}
