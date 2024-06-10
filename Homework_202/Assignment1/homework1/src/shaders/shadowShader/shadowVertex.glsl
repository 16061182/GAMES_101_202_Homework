attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;
attribute vec2 aTextureCoord;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uLightMVP;

varying highp vec3 vNormal; // mmc varing表示这是一个需要在fragment shader插值的顶点数据，应该是在shader里定义就行，在.js里没看见相关语句
varying highp vec2 vTextureCoord;

void main(void) {

  vNormal = aNormalPosition;
  vTextureCoord = aTextureCoord;

  gl_Position = uLightMVP * vec4(aVertexPosition, 1.0);
}