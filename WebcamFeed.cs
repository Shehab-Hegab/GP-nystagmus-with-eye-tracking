//////////using System;
//////////using System.IO;
//////////using System.Net.Sockets;
//////////using UnityEngine;

//////////public class WebcamVideoSender : MonoBehaviour
//////////{
//////////    public string serverIP = "127.0.0.1"; // IP address of Python server
//////////    public int serverPort = 5000;        // Port number of Python server

//////////    private TcpClient client;
//////////    private NetworkStream stream;
//////////    private WebCamTexture webcamTexture;
//////////    private Texture2D frameTexture;

//////////    void Start()
//////////    {
//////////        // Start the webcam
//////////        webcamTexture = new WebCamTexture();
//////////        frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
//////////        webcamTexture.Play();

//////////        // Connect to the Python server
//////////        try
//////////        {
//////////            client = new TcpClient(serverIP, serverPort);
//////////            stream = client.GetStream();
//////////            Debug.Log("Connected to the Python server.");
//////////        }
//////////        catch (Exception e)
//////////        {
//////////            Debug.LogError($"Error connecting to server: {e.Message}");
//////////        }
//////////    }

//////////    void Update()
//////////    {
//////////        if (client == null || !stream.CanWrite || !webcamTexture.isPlaying) return;

//////////        // Capture the current frame from the webcam
//////////        frameTexture.SetPixels(webcamTexture.GetPixels());
//////////        frameTexture.Apply();

//////////        // Encode the frame to JPEG format
//////////        byte[] imageData = frameTexture.EncodeToJPG();

//////////        // Send the size of the frame followed by the frame data
//////////        try
//////////        {
//////////            byte[] sizeData = BitConverter.GetBytes(imageData.Length);
//////////            stream.Write(sizeData, 0, sizeData.Length); // Send frame size
//////////            stream.Write(imageData, 0, imageData.Length); // Send frame data
//////////        }
//////////        catch (Exception e)
//////////        {
//////////            Debug.LogError($"Error sending data: {e.Message}");
//////////        }
//////////    }

//////////    private void OnApplicationQuit()
//////////    {
//////////        // Cleanup
//////////        if (webcamTexture != null)
//////////        {
//////////            webcamTexture.Stop();
//////////        }
//////////        if (stream != null) stream.Close();
//////////        if (client != null) client.Close();
//////////    }
//////////}

////////using System;
////////using System.IO;
////////using System.Net.Sockets;
////////using UnityEngine;

////////public class WebcamVideoSender : MonoBehaviour
////////{
////////    public string serverIP = "127.0.0.1"; // IP address of Python server
////////    public int serverPort = 5000;        // Port number of Python server

////////    private TcpClient client;
////////    private NetworkStream stream;
////////    private WebCamTexture webcamTexture;
////////    private Texture2D frameTexture;
////////    private string saveFolder;

////////    void Start()
////////    {
////////        // Start the webcam
////////        webcamTexture = new WebCamTexture();
////////        frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
////////        webcamTexture.Play();

////////        // Create a folder to save frames
////////        saveFolder = Path.Combine(Application.dataPath, "CapturedFrames");
////////        if (!Directory.Exists(saveFolder))
////////        {
////////            Directory.CreateDirectory(saveFolder);
////////        }

////////        // Connect to the Python server
////////        try
////////        {
////////            client = new TcpClient(serverIP, serverPort);
////////            stream = client.GetStream();
////////            Debug.Log("Connected to the Python server.");
////////        }
////////        catch (Exception e)
////////        {
////////            Debug.LogError($"Error connecting to server: {e.Message}");
////////        }
////////    }

////////    void Update()
////////    {
////////        if (!webcamTexture.isPlaying) return;

////////        // Capture the current frame from the webcam
////////        frameTexture.SetPixels(webcamTexture.GetPixels());
////////        frameTexture.Apply();

////////        // Add a timestamp to the frame
////////        AddTimestampToFrame();

////////        // Save the frame locally
////////        SaveFrameLocally();

////////        // Send the frame to the Python server
////////        SendFrameOverNetwork();
////////    }

////////    void AddTimestampToFrame()
////////    {
////////        // Generate a timestamp string
////////        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");

////////        // Create a new texture to draw the timestamp on
////////        Texture2D timestampedTexture = new Texture2D(frameTexture.width, frameTexture.height, TextureFormat.RGB24, false);
////////        timestampedTexture.SetPixels(frameTexture.GetPixels());
////////        timestampedTexture.Apply();

////////        // Draw the timestamp on the texture
////////        DrawTextOnTexture(timestampedTexture, timestamp, new Vector2(10, 10), Color.white, 20);

////////        // Replace the original frame texture with the timestamped one
////////        frameTexture = timestampedTexture;
////////    }

////////    void DrawTextOnTexture(Texture2D texture, string text, Vector2 position, Color color, int fontSize)
////////    {
////////        // Create a temporary RenderTexture
////////        RenderTexture renderTexture = RenderTexture.GetTemporary(texture.width, texture.height);
////////        Graphics.Blit(texture, renderTexture);

////////        // Set the active RenderTexture
////////        RenderTexture.active = renderTexture;

////////        // Create a new Texture2D to read the RenderTexture data
////////        Texture2D textTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false);
////////        textTexture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
////////        textTexture.Apply();

////////        // Draw the text on the texture
////////        GUIStyle style = new GUIStyle();
////////        style.fontSize = fontSize;
////////        style.normal.textColor = color;

////////        // Create a temporary GUI texture to draw the text
////////        GUI.skin.label.alignment = TextAnchor.UpperLeft;
////////        GUI.skin.label.fontSize = fontSize;
////////        GUI.skin.label.normal.textColor = color;

////////        // Draw the text on the texture
////////        RenderTexture.active = renderTexture;
////////        GUI.Label(new Rect(position.x, position.y, texture.width, texture.height), text, style);

////////        // Read the RenderTexture data back into the texture
////////        texture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0);
////////        texture.Apply();

////////        // Clean up
////////        RenderTexture.ReleaseTemporary(renderTexture);
////////    }

////////    void SaveFrameLocally()
////////    {
////////        // Generate a timestamp for the frame filename
////////        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
////////        string filePath = Path.Combine(saveFolder, $"frame_{timestamp}.png");

////////        // Encode the frame as a PNG and save it
////////        byte[] imageData = frameTexture.EncodeToPNG();
////////        File.WriteAllBytes(filePath, imageData);

////////        Debug.Log($"Frame saved locally at: {filePath}");
////////    }

////////    void SendFrameOverNetwork()
////////    {
////////        if (client == null || !stream.CanWrite) return;

////////        // Encode the frame as a JPEG
////////        byte[] imageData = frameTexture.EncodeToJPG();

////////        try
////////        {
////////            // Send the size of the frame first
////////            byte[] sizeData = BitConverter.GetBytes(imageData.Length);
////////            stream.Write(sizeData, 0, sizeData.Length);

////////            // Send the frame data
////////            stream.Write(imageData, 0, imageData.Length);
////////            Debug.Log("Frame sent to Python server.");
////////        }
////////        catch (Exception e)
////////        {
////////            Debug.LogError($"Error sending frame: {e.Message}");
////////        }
////////    }

////////    private void OnApplicationQuit()
////////    {
////////        // Cleanup
////////        if (webcamTexture != null)
////////        {
////////            webcamTexture.Stop();
////////        }
////////        if (stream != null) stream.Close();
////////        if (client != null) client.Close();
////////    }
////////}

//////using System;
//////using System.IO;
//////using System.Net.Sockets;
//////using UnityEngine;

//////public class WebcamVideoSender : MonoBehaviour
//////{
//////    public string serverIP = "127.0.0.1"; // IP address of Python server
//////    public int serverPort = 5000;        // Port number of Python server
//////    public string saveFolderName = "CapturedFrames"; // Folder name to save frames

//////    private TcpClient client;
//////    private NetworkStream stream;
//////    private WebCamTexture webcamTexture;
//////    private Texture2D frameTexture;
//////    private string saveFolderPath;

//////    void Start()
//////    {
//////        // Start the webcam
//////        webcamTexture = new WebCamTexture();
//////        frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
//////        webcamTexture.Play();

//////        // Create the folder to save frames
//////        saveFolderPath = Path.Combine(Application.dataPath, saveFolderName);
//////        if (!Directory.Exists(saveFolderPath))
//////        {
//////            Directory.CreateDirectory(saveFolderPath);
//////        }

//////        // Connect to the Python server
//////        try
//////        {
//////            client = new TcpClient(serverIP, serverPort);
//////            stream = client.GetStream();
//////            Debug.Log("Connected to the Python server.");
//////        }
//////        catch (Exception e)
//////        {
//////            Debug.LogError($"Error connecting to server: {e.Message}");
//////        }
//////    }

//////    void Update()
//////    {
//////        if (!webcamTexture.isPlaying) return;

//////        // Capture the current frame from the webcam
//////        frameTexture.SetPixels(webcamTexture.GetPixels());
//////        frameTexture.Apply();

//////        // Save the frame locally with a timestamped filename
//////        SaveFrameLocally();

//////        // Send the frame to the Python server
//////        SendFrameOverNetwork();
//////    }

//////    void SaveFrameLocally()
//////    {
//////        // Generate a timestamp for the filename
//////        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
//////        string filePath = Path.Combine(saveFolderPath, $"frame_{timestamp}.png");

//////        // Encode the frame as a PNG and save it
//////        byte[] imageData = frameTexture.EncodeToPNG();
//////        File.WriteAllBytes(filePath, imageData);

//////        Debug.Log($"Frame saved locally at: {filePath}");
//////    }

//////    void SendFrameOverNetwork()
//////    {
//////        if (client == null || !stream.CanWrite) return;

//////        // Encode the frame as a JPEG
//////        byte[] imageData = frameTexture.EncodeToJPG();

//////        try
//////        {
//////            // Send the size of the frame first
//////            byte[] sizeData = BitConverter.GetBytes(imageData.Length);
//////            stream.Write(sizeData, 0, sizeData.Length);

//////            // Send the frame data
//////            stream.Write(imageData, 0, imageData.Length);
//////            Debug.Log("Frame sent to Python server.");
//////        }
//////        catch (Exception e)
//////        {
//////            Debug.LogError($"Error sending frame: {e.Message}");
//////        }
//////    }

//////    private void OnApplicationQuit()
//////    {
//////        // Cleanup
//////        if (webcamTexture != null)
//////        {
//////            webcamTexture.Stop();
//////        }
//////        if (stream != null) stream.Close();
//////        if (client != null) client.Close();
//////    }
//////}

////using System;
////using System.IO;
////using System.Net.Sockets;
////using UnityEngine;

////public class WebcamVideoSender : MonoBehaviour
////{
////    public string serverIP = "127.0.0.1"; // IP address of Python server
////    public int serverPort = 5000;        // Port number of Python server
////    public string saveFolderName = "CapturedFrames"; // Folder name to save frames

////    private TcpClient client;
////    private NetworkStream stream;
////    private WebCamTexture webcamTexture;
////    private Texture2D frameTexture;
////    private string saveFolderPath;

////    void Start()
////    {
////        // Start the webcam
////        webcamTexture = new WebCamTexture();
////        frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
////        webcamTexture.Play();

////        // Create the folder to save frames
////        saveFolderPath = Path.Combine(Application.dataPath, saveFolderName);
////        if (!Directory.Exists(saveFolderPath))
////        {
////            Directory.CreateDirectory(saveFolderPath);
////        }

////        // Connect to the Python server
////        try
////        {
////            client = new TcpClient(serverIP, serverPort);
////            stream = client.GetStream();
////            Debug.Log("Connected to the Python server.");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"Error connecting to server: {e.Message}");
////        }
////    }

////    void Update()
////    {
////        if (!webcamTexture.isPlaying) return;

////        // Capture the current frame from the webcam
////        frameTexture.SetPixels(webcamTexture.GetPixels());
////        frameTexture.Apply();

////        // Save the frame locally with a timestamped filename
////        SaveFrameLocally();

////        // Send the frame to the Python server
////        SendFrameOverNetwork();
////    }

////    void SaveFrameLocally()
////    {
////        // Generate a timestamp for the filename
////        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
////        string filePath = Path.Combine(saveFolderPath, $"frame_{timestamp}.jpg");

////        // Encode the frame as a JPEG and save it
////        byte[] imageData = frameTexture.EncodeToJPG(); // Save as JPEG
////        File.WriteAllBytes(filePath, imageData);

////        Debug.Log($"Frame saved locally at: {filePath}");
////    }

////    void SendFrameOverNetwork()
////    {
////        if (client == null || !stream.CanWrite) return;

////        // Encode the frame as a JPEG
////        byte[] imageData = frameTexture.EncodeToJPG();

////        try
////        {
////            // Send the size of the frame first
////            byte[] sizeData = BitConverter.GetBytes(imageData.Length);
////            stream.Write(sizeData, 0, sizeData.Length);

////            // Send the frame data
////            stream.Write(imageData, 0, imageData.Length);
////            Debug.Log("Frame sent to Python server.");
////        }
////        catch (Exception e)
////        {
////            Debug.LogError($"Error sending frame: {e.Message}");
////        }
////    }

////    private void OnApplicationQuit()
////    {
////        // Cleanup
////        if (webcamTexture != null)
////        {
////            webcamTexture.Stop();
////        }
////        if (stream != null) stream.Close();
////        if (client != null) client.Close();
////    }
////}\\

//using System;
//using System.IO;
//using System.Net.Sockets;
//using UnityEngine;

//public class WebcamVideoSender : MonoBehaviour
//{
//    public string serverIP = "127.0.0.1"; // IP address of Python server
//    public int serverPort = 5000;        // Port number of Python server
//    public string saveFolderName = "CapturedFrames"; // Folder name to save frames

//    private TcpClient client;
//    private NetworkStream stream;
//    private WebCamTexture webcamTexture;
//    private Texture2D frameTexture;
//    private string saveFolderPath;

//    void Start()
//    {
//        // Start the webcam
//        webcamTexture = new WebCamTexture();
//        webcamTexture.Play();

//        // Initialize the texture to store the current frame
//        frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);

//        // Create the folder to save frames
//        saveFolderPath = Path.Combine(Application.dataPath, saveFolderName);
//        if (!Directory.Exists(saveFolderPath))
//        {
//            Directory.CreateDirectory(saveFolderPath);
//        }

//        // Connect to the Python server
//        try
//        {
//            client = new TcpClient(serverIP, serverPort);
//            stream = client.GetStream();
//            Debug.Log("Connected to the Python server.");
//        }
//        catch (Exception e)
//        {
//            Debug.LogError($"Error connecting to server: {e.Message}");
//        }
//    }

//    void Update()
//    {
//        if (!webcamTexture.isPlaying || !webcamTexture.didUpdateThisFrame) return;

//        // Capture the current frame from the webcam
//        frameTexture.SetPixels(webcamTexture.GetPixels());
//        frameTexture.Apply();

//        // Save the frame locally with a timestamped filename
//        string filePath = SaveFrameLocally();

//        // Send the frame to the Python server
//        SendFrameOverNetwork(filePath);
//    }

//    string SaveFrameLocally()
//    {
//        // Generate a timestamp for the filename
//        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
//        string filePath = Path.Combine(saveFolderPath, $"frame_{timestamp}.jpg");

//        // Encode the frame as a JPEG and save it
//        byte[] imageData = frameTexture.EncodeToJPG();
//        File.WriteAllBytes(filePath, imageData);

//        Debug.Log($"Frame saved locally at: {filePath}");
//        return filePath;
//    }

//    void SendFrameOverNetwork(string filePath)
//    {
//        if (client == null || !stream.CanWrite) return;

//        // Read the saved frame from disk
//        byte[] imageData = File.ReadAllBytes(filePath);

//        try
//        {
//            // Send the size of the frame first
//            byte[] sizeData = BitConverter.GetBytes(imageData.Length);
//            stream.Write(sizeData, 0, sizeData.Length);

//            // Send the frame data
//            stream.Write(imageData, 0, imageData.Length);
//            Debug.Log("Frame sent to Python server.");
//        }
//        catch (Exception e)
//        {
//            Debug.LogError($"Error sending frame: {e.Message}");
//        }
//    }

//    private void OnApplicationQuit()
//    {
//        // Cleanup
//        if (webcamTexture != null)
//        {
//            webcamTexture.Stop();
//        }
//        if (stream != null) stream.Close();
//        if (client != null) client.Close();
//    }
//}

using System;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class WebcamVideoSender : MonoBehaviour
{
    public string serverIP = "127.0.0.1"; // IP address of Python server
    public int serverPort = 5000;        // Port number of Python server
    public string saveFolderName = "CapturedFrames"; // Folder name to save frames

    private TcpClient client;
    private NetworkStream stream;
    private WebCamTexture webcamTexture;
    private Texture2D frameTexture;
    private string saveFolderPath;
    private bool isSending = false;

    void Start()
    {
        // Start the webcam
        webcamTexture = new WebCamTexture();
        webcamTexture.Play();

        // Initialize the texture to store the current frame
        frameTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);

        // Create the folder to save frames
        saveFolderPath = Path.Combine(Application.dataPath, saveFolderName);
        if (!Directory.Exists(saveFolderPath))
        {
            Directory.CreateDirectory(saveFolderPath);
        }

        // Connect to the Python server
        try
        {
            client = new TcpClient(serverIP, serverPort);
            stream = client.GetStream();
            Debug.Log("Connected to the Python server.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error connecting to server: {e.Message}");
        }
    }

    void Update()
    {
        if (!webcamTexture.isPlaying || !webcamTexture.didUpdateThisFrame) return;

        // Capture the current frame from the webcam
        frameTexture.SetPixels(webcamTexture.GetPixels());
        frameTexture.Apply();

        // Save the frame locally with a timestamped filename
        string filePath = SaveFrameLocally();

        // Send the frame to the Python server in a separate thread
        if (!isSending)
        {
            isSending = true;
            Thread sendThread = new Thread(() => SendFrameOverNetwork(filePath));
            sendThread.Start();
        }
    }

    string SaveFrameLocally()
    {
        // Generate a timestamp for the filename
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
        string filePath = Path.Combine(saveFolderPath, $"frame_{timestamp}.jpg");

        // Encode the frame as a JPEG and save it
        byte[] imageData = frameTexture.EncodeToJPG();
        File.WriteAllBytes(filePath, imageData);

        Debug.Log($"Frame saved locally at: {filePath}");
        return filePath;
    }

    void SendFrameOverNetwork(string filePath)
    {
        if (client == null || !stream.CanWrite) return;

        try
        {
            // Read the saved frame from disk
            byte[] imageData = File.ReadAllBytes(filePath);

            // Send the frame data directly (no frame size prefix)
            stream.Write(imageData, 0, imageData.Length);
            Debug.Log("Frame sent to Python server.");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending frame: {e.Message}");
        }
        finally
        {
            isSending = false; // Allow the next frame to be sent
        }
    }

    private void OnApplicationQuit()
    {
        // Cleanup
        if (webcamTexture != null)
        {
            webcamTexture.Stop();
        }
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
}