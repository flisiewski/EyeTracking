package com.agh.eyedetection;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;


import org.opencv.face.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;


public class MainActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    public static final int JAVA_DETECTOR = 0;

    private int openEyeLen;

    // matrix for zooming
    private Mat mZoomWindow;
    private Mat mZoomWindow2;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;

    private Point leftEyeLeftCorner;
    private Point leftEyeRightCorner;
    private Point rightEyeLeftCorner;
    private Point rightEyeRightCorner;

    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private File mCascadeFileEye;
    private File modelFile;
    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEye;


    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    double xCenter = -1;
    double yCenter = -1;

    Facemark fm;


    final static int MY_PERMISSIONS_REQUEST_LOCATION = 1;

    @Override
    protected void onStart() {
        super.onStart();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    2);


            // MY_PERMISSIONS_REQUEST_LOCATION is an
            // app-defined int constant. The callback method gets the
            // result of the request.
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    3);

        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    try {
                        fm = Face.createFacemarkLBF();

                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        // load cascade file from application resources
                        InputStream ise = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFileEye = new File(cascadeDirEye, "haarcascade_lefteye_2splits.xml");
                        FileOutputStream ose = new FileOutputStream(mCascadeFileEye);

                        while ((bytesRead = ise.read(buffer)) != -1) {
                            ose.write(buffer, 0, bytesRead);
                        }
                        ise.close();
                        ose.close();

                        InputStream isee = getResources().openRawResource(R.raw.lbfmodel);
                        File modelDir = getDir("cascade", Context.MODE_PRIVATE);
                        modelFile = new File(modelDir, "lbfmodel.yaml");
                        FileOutputStream osee = new FileOutputStream(modelFile);

                        while ((bytesRead = isee.read(buffer)) != -1) {
                            osee.write(buffer, 0, bytesRead);
                        }
                        isee.close();
                        osee.close();

                        fm.loadModel(modelFile.getAbsolutePath());

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mJavaDetectorEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
                        if (mJavaDetectorEye.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier for eye");
                            mJavaDetectorEye = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFileEye.getAbsolutePath());

                        cascadeDir.delete();
                        cascadeDirEye.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        mZoomWindow.release();
        mZoomWindow2.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }

        }

        if (mZoomWindow == null || mZoomWindow2 == null)
            CreateAuxiliaryMats();

        MatOfRect faces = new MatOfRect();

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        } else {
            Log.e(TAG, "Detection method is not selected!");
        }

        try {
            ArrayList<MatOfPoint2f> landmarks = new ArrayList<MatOfPoint2f>();
            fm.fit(mGray, faces, landmarks);

            // draw them
            for (int i = 0; i < landmarks.size(); i++) {

                MatOfPoint2f lm = landmarks.get(i);
                for (int j = 0; j < lm.rows(); j++) {
                    if (j == 36 || j == 39 || j == 42 || j == 45) {
                        double[] dp = lm.get(j, 0);
                        Point p = new Point(dp[0], dp[1]);
                        if (j == 36) {
                            leftEyeLeftCorner = p;
                        } else if (j == 39) {
                            leftEyeRightCorner = p;
                        } else if (j == 42) {
                            rightEyeLeftCorner = p;
                        } else {
                            rightEyeRightCorner = p;
                        }
                    }
                }

                openEyeLen = (int)(lm.get(37, 0)[1] - lm.get(41, 0)[1] +
                              lm.get(38, 0)[1] - lm.get(40, 0)[1] +
                              lm.get(43, 0)[1] - lm.get(47, 0)[1] +
                              lm.get(44, 0)[1] - lm.get(46, 0)[1]) / 4;
            }

        } catch (Exception ex) {

        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {

            xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
            yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;
            Point center = new Point(xCenter, yCenter);



            Rect r = facesArray[i];
            // compute the eye area

            // split it
            Rect eyearea_right = new Rect(r.x + r.width / 16,
                    (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            Rect eyearea_left = new Rect(r.x + r.width / 16
                    + (r.width - 2 * r.width / 16) / 2,
                    (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));


            EyeLookingVector left, right;
            right = get_template(mJavaDetectorEye, eyearea_right, 24, "right");
            left = get_template(mJavaDetectorEye, eyearea_left, 24, "left");

            if(right != null && left != null && left.awayFromCenter + right.awayFromCenter > 20 - openEyeLen/2){
                int newXDiff = (int) (right.endPoint.x - right.iris.x + left.endPoint.x - left.iris.x)/2;
                int newYDiff = (int) (right.endPoint.y - right.iris.y + left.endPoint.y - left.iris.y)/2;

                Point newLeftEnd = new Point(left.iris.x + newXDiff, left.iris.y + newYDiff);
                Point newRightEnd = new Point(right.iris.x + newXDiff, right.iris.y + newYDiff);

                if(Math.abs(newXDiff) < 50 && Math.abs(newYDiff) < 50 && (Math.abs(newXDiff) > 10 || Math.abs(newYDiff) > 10)) {
                    Imgproc.arrowedLine(mRgba, left.iris, newLeftEnd, new Scalar(255, 0, 0, 255), 5, 1, 0, 1);
                    Imgproc.arrowedLine(mRgba, right.iris, newRightEnd, new Scalar(255, 0, 0, 255), 5, 1, 0, 1);
                }
            }
        }

        return mRgba;
    }

    private void CreateAuxiliaryMats() {
        if (mGray.empty())
            return;

        int rows = mGray.rows();
        int cols = mGray.cols();

        if (mZoomWindow == null) {
            mZoomWindow = mRgba.submat(rows / 2 + rows / 10, rows, cols / 2
                    + cols / 10, cols);
            mZoomWindow2 = mRgba.submat(0, rows / 2 - rows / 10, cols / 2
                    + cols / 10, cols);
        }

    }

    private EyeLookingVector get_template(CascadeClassifier clasificator, Rect area, int size, String which) {
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        EyeLookingVector eyeLookingVector = null;

        clasificator.detectMultiScale(mROI, eyes, 1.15, 2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30),
                new Size());


        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length; ) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.4), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);

            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;

            Point leftEnd, rightEnd;
            double leftAwayFromCenter, rightAwayFromCenter;
            if (which.equals("right")) {
                Point center = new Point((leftEyeLeftCorner.x + leftEyeRightCorner.x) / 2.0, (leftEyeLeftCorner.y + leftEyeRightCorner.y) / 2.0  - 20 - openEyeLen/2.0);
                rightEnd = new Point(iris.x + (iris.x - center.x) * 2, iris.y + (iris.y - center.y) * 2);
                rightAwayFromCenter = Math.sqrt(Math.pow(center.x - rightEnd.x, 2) + Math.pow(center.y - rightEnd.y, 2));
                eyeLookingVector = new EyeLookingVector(iris, rightEnd, rightAwayFromCenter);
            } else {
                Point center = new Point((rightEyeLeftCorner.x + rightEyeRightCorner.x) / 2.0, (rightEyeLeftCorner.y + rightEyeRightCorner.y) / 2.0  - 20 - openEyeLen/2.0);
                leftEnd = new Point(iris.x + (iris.x - center.x) * 2, iris.y + (iris.y - center.y) * 2);
                leftAwayFromCenter = Math.sqrt(Math.pow(center.x - leftEnd.x, 2) + Math.pow(center.y - leftEnd.y, 2));
                eyeLookingVector = new EyeLookingVector(iris, leftEnd, leftAwayFromCenter);
            }

            return eyeLookingVector;
        }

        return eyeLookingVector;
    }
}