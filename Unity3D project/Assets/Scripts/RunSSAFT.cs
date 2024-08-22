using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using Lays = Unity.Sentis.Layers;
using System.IO;
using FF = Unity.Sentis.Functional;
using Unity.VisualScripting;
using UnityEngine.UIElements;
using Image = UnityEngine.UI.Image;
using UnityEngine.Android;
using System;

// Rotation speed in degrees per second




public class RunSSAFT : MonoBehaviour
{
    public Camera mainCamera;
    public List<string> ObjectLabels;
    public List<GameObject> ObjectsPrefs;
    public Dictionary<string, GameObject> spawnablePrefabs = new Dictionary<string, GameObject>();


    public float fixedDepth = -500.0f;  // Depth at which the object should float
    public Material blendMaterial;
    public RenderTexture someRenderTexture;

    // Drag the sentis file here
    public ModelAsset asset;
    const string modelName = "SSAFT.onnx";
    //const string videoName = "vid.mp4";
    // Link the classes.txt here:
    public TextAsset labelsAsset;
    // Create a Raw Image in the scene and link it here:
    public RawImage displayImage;
    // Link to a bounding box sprite or texture here:
    public Sprite borderSprite;
    public Texture2D borderTexture;
    // Link to the font for the labels:
    public Font font;

    const BackendType backend = BackendType.GPUCompute;

    private Transform displayLocation;
    private IWorker engine;
    private string[] labels;
    private RenderTexture targetRT;
    public WebCamTexture webcamTexture;


    //Image size for the model
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    private VideoPlayer video;


    List<GameObject> boxPool = new();
    List<GameObject> ObjectPool = new();

    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;
    int maxOutputBoxes = 64;
    public List<string> desiredClasses = new List<string> { "Coffee cup", "Computer mouse", "Computer keyboard",
        "Cutting board", "Scissors" };


    TensorFloat centersToCorners;
    //bounding box data
    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
        public float lastUpdatedTime;  // Timestamp of the last update
    }

    private void Awake()
    {
        displayImage.material = blendMaterial;




        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        //Parse neural net labels
        labels = labelsAsset.text.Split('\n');

        LoadModel();

        targetRT = new RenderTexture(imageWidth, imageHeight, 0);

        //Create image to display video
        displayLocation = displayImage.transform;
        SetupWebcam();


        SetupInput();

        if (borderSprite == null)
        {
            borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));
        }


    }

    void Start()
    {

        // Request camera permission for Android
        if (Application.platform == RuntimePlatform.Android)
        {
            if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
            {
                Permission.RequestUserPermission(Permission.Camera);
            }
        }

        for (int i = 0; i < ObjectLabels.Count; i++)
        {
            spawnablePrefabs.Add(ObjectLabels[i], ObjectsPrefs[i]);
        }


    }

    void SetupWebcam()
    {
        webcamTexture = new WebCamTexture();
        displayImage.texture = webcamTexture;
        webcamTexture.Play();
    }
    void LoadModel()
    {

        //Load model
        //var model1 = ModelLoader.Load(Path.Join(Application.streamingAssetsPath, modelName));
        var model1 = ModelLoader.Load(asset);

        centersToCorners = new TensorFloat(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        //Here we transform the output of the model1 by feeding it through a Non-Max-Suppression layer.
        var model2 = Functional.Compile(
               input =>
               {
                   var modelOutput = model1.Forward(input)[0];
                   var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);        //shape=(8400,4)
                   var allScores = modelOutput[0, 4.., ..];                         //shape=(80,8400)
                   var scores = FF.ReduceMax(allScores, 0) - scoreThreshold;        //shape=(8400)
                   var classIDs = FF.ArgMax(allScores, 0);                          //shape=(8400) 
                   var boxCorners = FF.MatMul(boxCoords, FunctionalTensor.FromTensor(centersToCorners));
                   var indices = FF.NMS(boxCorners, scores, iouThreshold);           //shape=(N)
                   var indices2 = indices.Unsqueeze(-1).BroadcastTo(new int[] { 4 });//shape=(N,4)
                   var coords = FF.Gather(boxCoords, 0, indices2);                  //shape=(N,4)
                   var labelIDs = FF.Gather(classIDs, 0, indices);                  //shape=(N)
                   return (coords, labelIDs);
               },
               InputDef.FromModel(model1)[0]
         );

        //Create engine to run model
        engine = WorkerFactory.CreateWorker(backend, model2);
    }

    void SetupInput()
    {
        video = gameObject.AddComponent<VideoPlayer>();
        video.renderMode = VideoRenderMode.APIOnly;
        video.source = VideoSource.Url;
        //video.url = Path.Join(Application.streamingAssetsPath, videoName);
        video.isLooping = true;
        video.Play();
    }



    private void Update()
    {
        if (webcamTexture.didUpdateThisFrame)
        {
            ExecuteML();
        }
        /*
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }*/
        blendMaterial.SetTexture("_MainTex", webcamTexture);
        blendMaterial.SetTexture("_OverlayTex", someRenderTexture);



    }

    public class BoundingBoxHolder : MonoBehaviour
    {
        public BoundingBox boundingBox;
    }


    public void ExecuteML()
    {
        ClearAnnotations();

        if (webcamTexture && webcamTexture.width > 100)
        {
            // Scaling might be necessary if the webcam's aspect ratio differs
            float aspect = webcamTexture.width * 1f / webcamTexture.height;
            Graphics.Blit(webcamTexture, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
            displayImage.texture = targetRT;
        }
        else return;

        using var input = TextureConverter.ToTensor(targetRT, imageWidth, imageHeight, 3);
        engine.Execute(input);

        var output = engine.PeekOutput("output_0") as TensorFloat;
        var labelIDs = engine.PeekOutput("output_1") as TensorInt;

        output.CompleteOperationsAndDownload();
        labelIDs.CompleteOperationsAndDownload();

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        int boxesFound = output.shape[0];
        //Draw the bounding boxes
        for (int n = 0; n < Mathf.Min(boxesFound, 200); n++)
        {
            string detectedLabel = labels[labelIDs[n]];
            if (desiredClasses.Contains(detectedLabel))
            {
                var box = new BoundingBox
                {
                    centerX = output[n, 0] * scaleX - displayWidth / 2,
                    centerY = output[n, 1] * scaleY - displayHeight / 2,
                    width = output[n, 2] * scaleX,
                    height = output[n, 3] * scaleY,
                    label = labels[labelIDs[n]],
                };
                DrawBox(box, n, displayHeight * 0.05f);
            }



        }
    }


    Vector3 CalculateScaleAndDepth(float width, float height)
    {
        // Constants to define how scale and depth relate to the bounding box size
        float baseScale = 0.5f;  // Base scale factor
        float depthFactor = 0.1f;  // Factor to calculate depth from size

        // Calculate scale as a function of the average of width and height
        float scale = baseScale * (width + height) / 2;

        // Calculate depth inversely proportional to the size of the bounding box
        float depth = fixedDepth - depthFactor * (width + height);

        return new Vector3(scale, scale, depth);
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        GameObject panel;
        GameObject object3d;

        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
            object3d = ObjectPool[id];
            object3d.SetActive(true);
            //EnsureRotationScript(object3d);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
            object3d = CreateNewObject3d(box);
            
        }
   

        object3d.SetActive(true);

        // Update object3d to correct prefab each time
        if (spawnablePrefabs.ContainsKey(box.label))
        {
            GameObject correctPrefab = spawnablePrefabs[box.label];
            if (object3d.name != correctPrefab.name + "(Clone)")  // Check if correct prefab is already used
            {
                Destroy(object3d);  // Remove wrong prefab
                object3d = Instantiate(correctPrefab, displayLocation);  // Create correct prefab
                ObjectPool[id] = object3d;  // Update ObjectPool with correct object
            }
        }
        if (!spawnablePrefabs.ContainsKey(box.label))
        {
            Debug.LogError("Prefab not found for label: " + box.label);
            return;  // Skip spawning if prefab is missing
        }

        float verticalOffset = (box.height * 0.5f) + 65;
        Vector3 scaleAndDepth = CalculateScaleAndDepth(box.width, box.height);
        // Set box and object position and size
        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);
        object3d.transform.localPosition = new Vector3(box.centerX, -box.centerY + verticalOffset, -500);  // Ensure it's visible and at correct depth
        object3d.transform.localScale = new Vector3(scaleAndDepth.x, scaleAndDepth.x, scaleAndDepth.x);

        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        // Set label text
        Text label = panel.GetComponentInChildren<Text>();
        label.text = box.label;
        label.fontSize = (int)fontSize;

    }

    public GameObject CreateNewBox(Color color)
    {
        //Create the box and set image

        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        UnityEngine.UI.Image img = panel.AddComponent<UnityEngine.UI.Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        //Create the label

        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);


        return panel;
    }


    public GameObject CreateNewObject3d(BoundingBox box)
    {
        //Create the box and set image

        var object3d = new GameObject();

        Debug.Log(box.label);

        object3d = Instantiate(spawnablePrefabs[box.label]);

        object3d.transform.SetParent(displayLocation, false);

        ObjectPool.Add(object3d);

        return object3d;
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
        foreach (var obj in ObjectPool)
        {
            obj.SetActive(false);
        }
    }

    private void OnDestroy()
    {
        // Dispose resources and stop webcam
        webcamTexture.Stop();
        centersToCorners?.Dispose();
        engine?.Dispose();
    }
}