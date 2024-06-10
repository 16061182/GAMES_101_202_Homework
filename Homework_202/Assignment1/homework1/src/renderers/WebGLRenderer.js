class WebGLRenderer {
    meshes = [];
    shadowMeshes = [];
    lights = [];

    constructor(gl, camera) {
        this.gl = gl;
        this.camera = camera;
    }

    addLight(light) {
        this.lights.push({
            entity: light,
            meshRender: new MeshRender(this.gl, light.mesh, light.mat)
        });
    }
    addMeshRender(mesh) { this.meshes.push(mesh); } // mmc loadobj调用，MeshRender类型
    addShadowMeshRender(mesh) { this.shadowMeshes.push(mesh); } // mmc loadobj调用，MeshRender类型

    //Edit Start 添加time, deltaime参数
    render(time, deltaime) {
    //Edit End
        const gl = this.gl;

        gl.clearColor(0.0, 0.0, 0.0, 1.0); // Clear to black, fully opaque
        gl.clearDepth(1.0); // Clear everything
        gl.enable(gl.DEPTH_TEST); // Enable depth testing // mmc webgl开启深度检测方法。shadow pass（渲染至shadow map）肯定也需要开启深度检测，于是寻找开启深度检测的位置，找到这里
        gl.depthFunc(gl.LEQUAL); // Near things obscure far things

        console.assert(this.lights.length != 0, "No light");
        //console.assert(this.lights.length == 1, "Multiple lights"); //取消多光源检测

        //Edit Start 角色旋转，地面不转(用顶点数筛选)
        for (let i = 0; i < this.meshes.length; i++) {
            if(this.meshes[i].mesh.count > 10)
            {
                this.meshes[i].mesh.transform.rotate[1] = this.meshes[i].mesh.transform.rotate[1] + degrees2Radians(10) * deltaime;
            }
        }
        /* mmc
        这里另外补充一个小说明，WebGLRenderer里包含了meshes和shadowMeshes两个数组字段，我们需要对他们都进行渲染，而我们这里只修改了meshes的旋转信息，
        是因为meshes和shadowMeshes并不是Mesh本身，而是两组MeshRender，这里命名不是很准确，MeshRender里才包含了真正的Mesh实例，
        而这两组MeshRender其实是指向同一组Mesh，所以我们只需要对其中一组进行修改即可。
        */
        //Edit End

        for (let l = 0; l < this.lights.length; l++) {

            //Edit Start 切换光源时，对当前光源的shadowmap的framebuffer做一些清理操作
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.lights[l].entity.fbo); // 绑定到当前光源的framebuffer
            gl.clearColor(1.0, 1.0, 1.0, 1.0); // shadowmap默认白色（无遮挡），解决地面边缘产生阴影的问题（因为地面外采样不到，默认值为0会认为是被遮挡） // mmc 采样shadow map采样不到就填成了默认值0？
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); // 清除shadowmap上一帧的颜色、深度缓存，否则会一直叠加每一帧的结果
            //Edit End

            // Draw light
            // TODO: Support all kinds of transform
            //Edit Start 灯光围绕原点旋转
            let lightRotateSpped = [10, 80]
            let lightPos = this.lights[l].entity.lightPos;
            lightPos = vec3.rotateY(lightPos, lightPos, this.lights[l].entity.focalPoint, degrees2Radians(lightRotateSpped[l]) * deltaime);
            this.lights[l].entity.lightPos = lightPos; //给DirectionalLight的lightPos赋值新的位置，CalcLightMVP计算LightMVP需要用到
            this.lights[l].meshRender.mesh.transform.translate = lightPos;
            //Edit End
            this.lights[l].meshRender.draw(this.camera);
            

            // Shadow pass
            if (this.lights[l].entity.hasShadowMap == true) {
                for (let i = 0; i < this.shadowMeshes.length; i++) {
                    if(this.shadowMeshes[i].material.lightIndex != l)
                        continue;// 是当前光源的材质才绘制，否则跳过
                    // Edit Start 每帧更新shader中uniforms的LightMVP
                    this.gl.useProgram(this.shadowMeshes[i].shader.program.glShaderProgram);
                    let translation = this.shadowMeshes[i].mesh.transform.translate;
                    let rotation = this.shadowMeshes[i].mesh.transform.rotate;
                    let scale = this.shadowMeshes[i].mesh.transform.scale;
                    let lightMVP = this.lights[l].entity.CalcLightMVP(translation, rotation, scale); // mmc lightMVP是正交投影
                    this.shadowMeshes[i].material.uniforms.uLightMVP = { type: 'matrix4fv', value: lightMVP }; // mmc 每帧更新lightmvp，以支持模型旋转，下同
                    // Edit End
                    this.shadowMeshes[i].draw(this.camera);
                }
            }

            // Edit Start 非第一个光源Pass时进行一些设置（Base Pass和Additional Pass区分）
            if(l != 0)
            {
                // 开启混合，把Additional Pass混合到Base Pass结果上，否则会覆盖Base Pass的渲染结果
                gl.enable(gl.BLEND);
                gl.blendFunc(gl.ONE, gl.ONE);
            }
            // Edit End

            // Camera pass
            for (let i = 0; i < this.meshes.length; i++) {
                if(this.meshes[i].material.lightIndex != l)
                    continue;// 是当前光源的材质才绘制，否则跳过
                this.gl.useProgram(this.meshes[i].shader.program.glShaderProgram);
                // Edit Start 每帧更新shader中uniforms参数
                // this.gl.uniform3fv(this.meshes[i].shader.program.uniforms.uLightPos, this.lights[l].entity.lightPos); //这里改用下面写法
                let translation = this.meshes[i].mesh.transform.translate;
                let rotation = this.meshes[i].mesh.transform.rotate;
                let scale = this.meshes[i].mesh.transform.scale;
                let lightMVP = this.lights[l].entity.CalcLightMVP(translation, rotation, scale);
                this.meshes[i].material.uniforms.uLightMVP = { type: 'matrix4fv', value: lightMVP }; // mmc lightMVP是正交投影；matrix在内存里是按行排的，uLightMVP[0][1][2][3]是第一行
                this.meshes[i].material.uniforms.uLightPos = { type: '3fv', value: this.lights[l].entity.lightPos }; // 光源方向计算、光源强度衰减
                // Edit End
                this.meshes[i].draw(this.camera);
            }

            // Edit Start 还原Additional Pass的设置
            gl.disable(gl.BLEND);
            // Edit End
        }
    }
}