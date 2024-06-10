let precomputeLT = [];
let precomputeL = [];
var cameraPosition = [0, 0, 100];

var envmap = [
	'assets/cubemap/GraceCathedral',
	'assets/cubemap/Indoor',
	'assets/cubemap/Skybox',
	// Edit Start
	'assets/cubemap/CornellBox',
	// Edit End
];

var guiParams = {
	envmapId: 0
}

var cubeMaps = [];

//生成的纹理的分辨率，纹理必须是标准的尺寸 256*256 1024*1024  2048*2048
var resolution = 2048;

let envMapPass = null;

GAMES202Main();

async function GAMES202Main() {
	// Init canvas and gl
	const canvas = document.querySelector('#glcanvas');
	canvas.width = window.screen.width;
	canvas.height = window.screen.height;
	const gl = canvas.getContext('webgl');
	if (!gl) {
		alert('Unable to initialize WebGL. Your browser or machine may not support it.');
		return;
	}

	// Add camera
	const camera = new THREE.PerspectiveCamera(75, gl.canvas.clientWidth / gl.canvas.clientHeight, 1e-2, 1000);
	camera.position.set(cameraPosition[0], cameraPosition[1], cameraPosition[2]);

	// Add resize listener
	function setSize(width, height) {
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
	}
	setSize(canvas.clientWidth, canvas.clientHeight);
	window.addEventListener('resize', () => setSize(canvas.clientWidth, canvas.clientHeight));

	// Add camera control
	const cameraControls = new THREE.OrbitControls(camera, canvas);
	cameraControls.enableZoom = true;
	cameraControls.enableRotate = true;
	cameraControls.enablePan = true;
	cameraControls.rotateSpeed = 0.3;
	cameraControls.zoomSpeed = 1.0;
	cameraControls.panSpeed = 0.8;
	cameraControls.target.set(0, 0, 0);

	// Add renderer
	const renderer = new WebGLRenderer(gl, camera);

	// Add lights
	// light - is open shadow map == false
	let lightPos = [0, 10000, 0];
	let lightRadiance = [1, 0, 0];
	const pointLight = new PointLight(lightRadiance, lightPos, false, renderer.gl);
	renderer.addLight(pointLight);

	// Add shapes
	let skyBoxTransform = setTransform(0, 50, 50, 150, 150, 150);
	let boxTransform = setTransform(0, 0, 0, 200, 200, 200);
	let box2Transform = setTransform(0, -10, 0, 20, 20, 20);

	for (let i = 0; i < envmap.length; i++) {
		let urls = [
			envmap[i] + '/posx.jpg',
			envmap[i] + '/negx.jpg',
			envmap[i] + '/posy.jpg',
			envmap[i] + '/negy.jpg',
			envmap[i] + '/posz.jpg',
			envmap[i] + '/negz.jpg',
		];
		cubeMaps.push(new CubeTexture(gl, urls))
		await cubeMaps[i].init();
	}
	// load skybox
	loadOBJ(renderer, 'assets/testObj/', 'testObj', 'SkyBoxMaterial', skyBoxTransform);

	// file parsing
	for (let i = 0; i < envmap.length; i++) {

		let val = '';
		await this.loadShaderFile(envmap[i] + "/transport.txt").then(result => {
			val = result;
		});

		let preArray = val.split(/[(\r\n)\r\n' ']+/); // mmc 按照空格把整个文件里的内容分割
		let lineArray = [];
		precomputeLT[i] = []
		for (let j = 1; j <= Number(preArray.length) - 2; j++) { // mmc 掐头去尾是因为第一个元素是顶点数量（fout << mesh->getVertexCount() << std::endl;），最后一个元素是空字符串
			precomputeLT[i][j - 1] = Number(preArray[j])
		}
		/* mmc precomputeLT的数据格式
		二维数组
		第一维：4个cubemap
		第二维：所有数据存成一个一维数组。数据每27个为一组，对应一个面，三个顶点各9个sh系数
		 */
		
		await this.loadShaderFile(envmap[i] + "/light.txt").then(result => {
			val = result;
		});

		precomputeL[i] = val.split(/[(\r\n)\r\n]+/);
		precomputeL[i].pop(); // mmc 删除最后一个空字符串
		for (let j = 0; j < 9; j++) { // mmc 9个sh系数（每个系数三通道）
			lineArray = precomputeL[i][j].split(' '); // mmc rgb三通道
			for (let k = 0; k < 3; k++) {
				lineArray[k] = Number(lineArray[k]); // mmc 字符串转数字
			}
			precomputeL[i][j] = lineArray;
		}
		/* mmc precomputeL的数据格式
		三维数组
		第一维：4个cubemap
		第二维：9个sh系数，前三阶
		第三维：3个通道，rgb
		 */
	}

	// TODO: load model - Add your Material here
	// loadOBJ(renderer, 'assets/bunny/', 'bunny', 'addYourPRTMaterial', boxTransform);
	// loadOBJ(renderer, 'assets/bunny/', 'bunny', 'addYourPRTMaterial', box2Transform);

	// Edit Start
	let maryTransform = setTransform(0, -35, 0, 20, 20, 20);
	loadOBJ(renderer, 'assets/mary/', 'mary', 'PRTMaterial', maryTransform); // mmc 设置使用PRTMaterial
	// Edit End

	function createGUI() {
		const gui = new dat.gui.GUI();
		const panelModel = gui.addFolder('Switch Environemtn Map');
		// Edit Start
		panelModel.add(guiParams, 'envmapId', { 'GraceGathedral': 0, 'Indoor': 1, 'Skybox': 2, 'CornellBox': 3}).name('Envmap Name');
		// Edit End
		panelModel.open();
	}

	createGUI();

	function mainLoop(now) {
		cameraControls.update();

		renderer.render();

		requestAnimationFrame(mainLoop);
	}
	requestAnimationFrame(mainLoop);
}

function setTransform(t_x, t_y, t_z, s_x, s_y, s_z) {
	return {
		modelTransX: t_x,
		modelTransY: t_y,
		modelTransZ: t_z,
		modelScaleX: s_x,
		modelScaleY: s_y,
		modelScaleZ: s_z,
	};
}