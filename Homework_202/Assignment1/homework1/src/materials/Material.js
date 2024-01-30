class Material {
    #flatten_uniforms;
    #flatten_attribs;
    #vsSrc;
    #fsSrc;
    // Uniforms is a map, attribs is a Array
    //Edit Start 添加lightIndex参数
    constructor(uniforms, attribs, vsSrc, fsSrc, frameBuffer, lightIndex) {
    //Edit End 
        this.uniforms = uniforms;
        this.attribs = attribs;
        this.#vsSrc = vsSrc;
        this.#fsSrc = fsSrc;
        
        this.#flatten_uniforms = ['uViewMatrix','uModelMatrix', 'uProjectionMatrix', 'uCameraPos', 'uLightPos'];
        // mmc 'uViewMatrix','uModelMatrix', 'uProjectionMatrix', 'uCameraPos'是在this.shadowMeshes[i].draw 或 this.meshes[i].draw -> meshrender.bindCameraParameters里绑的
        // mmc 'uLightPos'是现在webglrenderer里手动添加一个uniform（this.meshes[i].material.uniforms.uLightPos = { type: '3fv', value: this.lights[l].entity.lightPos };），然后this.meshes[i].draw -> meshrender.bindMaterialParameters里绑的
        for (let k in uniforms) {
            this.#flatten_uniforms.push(k); // mmc 这里添加的是uniforms的key，也就是像'uLightIntensity'这种字符串，不是字典，this.#flatten_uniforms还是一个字符串列表
        }
        this.#flatten_attribs = attribs;

        this.frameBuffer = frameBuffer;
        //Edit Start 添加lightIndex字段
        this.lightIndex = lightIndex;
        //Edit End
    }

    setMeshAttribs(extraAttribs) {
        for (let i = 0; i < extraAttribs.length; i++) {
            this.#flatten_attribs.push(extraAttribs[i]);
        }
    }

    compile(gl) {
        return new Shader(gl, this.#vsSrc, this.#fsSrc,
            {
                uniforms: this.#flatten_uniforms, // mmc 这里只声明了uniforms的名字，没有传值，因为this.#flatten_uniforms是一个字符串数组
                attribs: this.#flatten_attribs // mmc 同理，只有名字
            });
    }
}