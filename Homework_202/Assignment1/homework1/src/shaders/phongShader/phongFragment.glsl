#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 50
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

//Edit Start
#define SHADOW_MAP_SIZE 2048.
#define FILTER_RADIUS 10.
#define FRUSTUM_SIZE 400. // mmc 近平面的尺寸，pcss时认为是shadow map的“尺寸”（单位非像素，注意是正方形）
#define NEAR_PLANE 0.01 // mmc pcss时，认为shadow map在近平面上
#define LIGHT_WORLD_SIZE 5. // mmc 自定义的光源大小
#define LIGHT_SIZE_UV LIGHT_WORLD_SIZE / FRUSTUM_SIZE // mmc 光源在ShadowMap上的UV单位大小
//Edit End

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    /* mmc
    随着i增大，radius不断增大，从0.02到1（NUM_SAMPLES为50）
    最后一个的时候radius是1，所以最后一个值是(cos(angle), sin(angle))，在半径为1的圆【上】
    因此poissonDisk[i]的所有点分布在半径为1的圆【内】
    */
    radius += radiusStep; // mmc radius从0.02到1（NUM_SAMPLES为50）
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

//Edit Start
//自适应Shadow Bias算法 https://zhuanlan.zhihu.com/p/370951892
float getShadowBias(float c, float filterRadiusUV){
  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(uLightPos - vFragPos);
  float fragSize = (1. + ceil(filterRadiusUV)) * (FRUSTUM_SIZE / SHADOW_MAP_SIZE / 2.);
  return max(fragSize, fragSize * (1.0 - dot(normal, lightDir))) * c;
}
//Edit End

//Edit Start
float useShadowMap(sampler2D shadowMap, vec4 shadowCoord, float biasC, float filterRadiusUV){
  float shadow_depth = unpack(texture2D(shadowMap, shadowCoord.xy));
  float cur_depth = shadowCoord.z;
  float bias = getShadowBias(biasC, filterRadiusUV);
  if(cur_depth - bias >= shadow_depth + EPS){
    return 0.;
  }
  else{
    return 1.0;
  }
}
//Edit End

//Edit Start
float PCF(sampler2D shadowMap, vec4 coords, float biasC, float filterRadiusUV) { // mmc filterRadiusUV 表示查询shadowmap的范围大小，例如shadow map是2k贴图，查询范围是10*10像素，那么filterRadiusUV为10. / 2048.
  //uniformDiskSamples(coords.xy);
  poissonDiskSamples(coords.xy); //使用xy坐标作为随机种子生成
  float visibility = 0.0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    vec2 offset = poissonDisk[i] * filterRadiusUV; // mmc offset是0到1的uv空间的值（coord也是0到1的uv空间的值）；【感觉】poissonDisk[i].xy均在-0.5到0.5之间比较合理（原点为中心边长为1的正方形），然而实际上poissonDisk[i].xy的值可能取到(0.707,0.707)（原点为圆心半径为1的圆）
    float noshadow = useShadowMap(shadowMap, coords + vec4(offset, 0., 0.), biasC, filterRadiusUV); // mmc 原版这里写错了
    if (noshadow != 0.0) {
      visibility++;
    }
  }
  return visibility / float(NUM_SAMPLES);
}
//Edit End

//Edit Start
float findBlocker(sampler2D shadowMap, vec2 uv, float zReceiver) {
  int blockerNum = 0;
  float blockDepth = 0.;

  float posZFromLight = vPositionFromLight.z; // mmc 要用世界空间下的深度，因为使用shadowCoord的深度无法体现光源到shadowmap的距离

  float searchRadius = LIGHT_SIZE_UV * (posZFromLight - NEAR_PLANE) / posZFromLight;
  /*mmc
    searchRadius / LIGHT_SIZE_UV = (posZFromLight - NEAR_PLANE) / posZFromLight
    shadowmap上的投影边长(shadowmap UV空间) / light的边长(shadowmap UV空间) = shading point到shadowmap的深度 / shading point到光源的深度
  */

  poissonDiskSamples(uv);
  for(int i = 0; i < NUM_SAMPLES; i++){
    float shadowDepth = unpack(texture2D(shadowMap, uv + poissonDisk[i] * searchRadius));
    if(zReceiver > shadowDepth){
      blockerNum++;
      blockDepth += shadowDepth;
    }
  }

  if(blockerNum == 0)
    return -1.;
  else
    return blockDepth / float(blockerNum);
}
//Edit End

//Edit Start
float PCSS(sampler2D shadowMap, vec4 coords, float biasC){
  float zReceiver = coords.z;

  // STEP 1: avgblocker depth 
  float avgBlockerDepth = findBlocker(shadowMap, coords.xy, zReceiver);

  if(avgBlockerDepth < -EPS)
    return 1.0;

  // STEP 2: penumbra size
  float penumbra = (zReceiver - avgBlockerDepth) * LIGHT_SIZE_UV / avgBlockerDepth;
  /*mmc
    penumbra / LIGHT_SIZE_UV = (zReceiver - avgBlockerDepth) / avgBlockerDepth
    半影区域的长度(shadowmap UV空间) / light的边长(shadowmap UV空间) = shading point到平均遮挡物的深度 / 平均遮挡物的深度
  */
  float filterRadiusUV = penumbra;

  // STEP 3: filtering
  return PCF(shadowMap, coords, biasC, filterRadiusUV);
}
//Edit End

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {
  //Edit Start
  //vPositionFromLight为光源空间下投影的裁剪坐标，除以w结果为NDC坐标
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w; // mmc uLightMVP用的正交投影，这步冗余，vPositionFromLight.w一定是1
  //把[-1,1]的NDC坐标转换为[0,1]的坐标 // mmc正交投影的结果是在-1到1之间的
  shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0; // mmc 注意这个值在0到1的uv空间，准确地说是【shadow map的UV空间】，shadow map上存的深度就是0到1的uv空间的深度（详见shadowFragment.glsl）

  float visibility = 1.;

  // 无PCF时的Shadow Bias
  float nonePCFBiasC = .4;
  // 有PCF时的Shadow Bias
  float pcfBiasC = 0.2;
  // PCF的采样范围，因为是在Shadow Map上采样，需要除以Shadow Map大小，得到uv坐标上的范围
  float filterRadiusUV = FILTER_RADIUS / SHADOW_MAP_SIZE;

  // 硬阴影无PCF，最后参数传0
  // mmc 注意float字面量后面别加f，和hlsl不一样写了f会变成不合法值直接解释为0
//  visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0), nonePCFBiasC, 0.);
//  visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0), pcfBiasC, filterRadiusUV);
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0), pcfBiasC);

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
  //Edit End
}

// mmc 每次更改代码之后，为了清除浏览器缓存，重新换一个新的端口号启动本地服务器，例如`http-server . -p 8001`
