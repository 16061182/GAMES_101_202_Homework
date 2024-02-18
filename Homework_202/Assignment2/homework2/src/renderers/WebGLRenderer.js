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
    addMeshRender(mesh) { this.meshes.push(mesh); }
    addShadowMeshRender(mesh) { this.shadowMeshes.push(mesh); }

    render() {
        const gl = this.gl;

        gl.clearColor(0.0, 0.0, 0.0, 1.0); // Clear to black, fully opaque
        gl.clearDepth(1.0); // Clear everything
        gl.enable(gl.DEPTH_TEST); // Enable depth testing
        gl.depthFunc(gl.LEQUAL); // Near things obscure far things

        console.assert(this.lights.length != 0, "No light");
        console.assert(this.lights.length == 1, "Multiple lights");

        const timer = Date.now() * 0.0001;

        for (let l = 0; l < this.lights.length; l++) {
            // Draw light
            this.lights[l].meshRender.mesh.transform.translate = this.lights[l].entity.lightPos;
            this.lights[l].meshRender.draw(this.camera);

            // Shadow pass
            if (this.lights[l].entity.hasShadowMap == true) {
                for (let i = 0; i < this.shadowMeshes.length; i++) {
                    this.shadowMeshes[i].draw(this.camera);
                }
            }

            // Camera pass
            for (let i = 0; i < this.meshes.length; i++) {
                this.gl.useProgram(this.meshes[i].shader.program.glShaderProgram);
                this.gl.uniform3fv(this.meshes[i].shader.program.uniforms.uLightPos, this.lights[l].entity.lightPos);

                for (let k in this.meshes[i].material.uniforms) {

                    let cameraModelMatrix = mat4.create();
                    // Edit Start
                    mat4.fromRotation(cameraModelMatrix, timer * 10, [0, 1, 0]); // mmc 【bonus】旋转时打开
                    // Edit End

                    if (k == 'uMoveWithCamera') { // The rotation of the skybox
                        gl.uniformMatrix4fv(
                            this.meshes[i].shader.program.uniforms[k],
                            false,
                            cameraModelMatrix);
                    }

                    // Bonus - Fast Spherical Harmonic Rotation
                    let precomputeL_RGBMat3 = getRotationPrecomputeL(precomputeL[guiParams.envmapId], cameraModelMatrix);  // mmc 【bonus】旋转时打开
                    
                    // Edit Start
                    // let Mat3Value = getMat3ValueFromRGB(precomputeL[guiParams.envmapId])  // mmc 【bonus】旋转时注释 // guiParams.envmapId是环境贴图id，就是那四个，cornell box之类的
                    /* mmc
                    * precomputeL[guiParams.envmapId]第一维是9个sh系数，第二维是rgb3个通道
                    * getMat3ValueFromRGB手动给这个矩阵做了一个转置
                    * Mat3Value第一维是3通道，第二维是9个sh系数
                    */

                    let Mat3Value = getMat3ValueFromRGB(precomputeL_RGBMat3);  // mmc 【bonus】旋转时打开
                    for(let j = 0; j < 3; j++){
                        if (k == 'uPrecomputeL['+j+']') { // mmc uPrecomputeL在shader里是个数组
                            gl.uniformMatrix3fv(
                                this.meshes[i].shader.program.uniforms[k], // mmc 类型是WebGLUniformLocation，uniform名到uniform location的映射（即this.meshes[i].shader.program.uniforms）是在shader的构造函数里用addShaderLocations函数创建的
                                false,
                                Mat3Value[j]);
                        }
                    }
                    // Edit End
                }

                this.meshes[i].draw(this.camera);
            }
        }

    }
}