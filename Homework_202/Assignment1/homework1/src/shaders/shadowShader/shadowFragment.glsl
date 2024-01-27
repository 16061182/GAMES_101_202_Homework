#ifdef GL_ES
precision mediump float;
#endif

uniform vec3 uLightPos;
uniform vec3 uCameraPos;

varying highp vec3 vNormal;
varying highp vec2 vTextureCoord;

vec4 pack (float depth) { // mmc 这个函数的功能是把一个[0, 1)的32位float编码成8bit RGBA四通道
    const vec4 bitShift = vec4(1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0);
    const vec4 bitMask = vec4(1.0/256.0, 1.0/256.0, 1.0/256.0, 0.0);
    vec4 rgbaDepth = fract(depth * bitShift); // mmc fract取小数，假设用二进制表示的输入为0.1111···（32个1），那么rgbaDepth.x为0.1111···（32个1），rgbaDepth.y为0.1111····（24个1，它的小数部分实际上是原数字的小数部分的第9到第32位），后面类推
    rgbaDepth -= rgbaDepth.gbaa * bitMask; // mmc 上一步得到的结果中，x有32位有效数字（第1到第32位），y有24位（第9到第32位），z有16位（第17到第32位），a有8位（第25到第32位），实际上四个通道各保留8位有效数字即可，这一步就是把每通道8位之后的小数去掉，变成rgbaDepth.xyza分别是第1到8、第9到16、第17到第24、第25到第32
    return rgbaDepth;
}

void main(){

  //gl_FragColor = vec4( 1.0, 0.0, 0.0, gl_FragCoord.z);
  gl_FragColor = pack(gl_FragCoord.z); // mmc gl_FragCoord是屏幕空间坐标，x：[0, ScreenWidth]，y：[0, ScreenHeight]，z：[0, 1] // 注意shadowmap中存的深度值是0到1的
  // mmc 这里我们使用了z值，能直接表示片元的深度关系，至于此深度值是否线性，取决于投影矩阵，因为平行光使用的是正交矩阵，所以gl_FragCoord.z取得的深度值是线性的。
  // mmc 而gl_FragCoord.w则是裁剪空间坐标中w的倒数。
  // mmc 这个pack函数不支持z值为1.0
}