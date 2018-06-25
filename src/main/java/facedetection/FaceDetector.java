package facedetection;

import java.io.File;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

/**
 * Detect faces.
 */
public class FaceDetector {

	 private boolean lbpClassifier = true;
	 private boolean haarClassifier = false;
//	private boolean lbpClassifier = false;
//	private boolean haarClassifier = true;
	private CascadeClassifier faceCascade;

	public FaceDetector() {
		this.faceCascade = new CascadeClassifier();
		String classifier = null;

		if (this.lbpClassifier) {
			// 2,5,7
			classifier = "/lbpcascades/lbpcascade_frontalface.xml";

			// not much
			// classifier="/lbpcascades/lbpcascade_profileface.xml";
//			classifier="/lbpcascades/lbpcascade_frontalcatface.xml";

			classifier = "/lbpcascades/lbpcascade_frontalface_improved.xml";
		}
		if (this.haarClassifier) {

			// 2
			classifier = "/haarcascades/haarcascade_profileface.xml";

			// 1,2,5,7
			// classifier = "/haarcascades/haarcascade_frontalcatface_extended.xml";
			// classifier = "/haarcascades/haarcascade_frontalface_default.xml";

			// 1,2,5,7
			// classifier = "/haarcascades/haarcascade_frontalface_alt.xml";
			// classifier = "/haarcascades/haarcascade_eye.xml";

			// 2,5
			// classifier = "/haarcascades/haarcascade_frontalface_alt_tree.xml";

			// 1,2,5,7
			// classifier = "/haarcascades/haarcascade_frontalface_alt2.xml";
		}

		File auxFile = new File(getClass().getResource(classifier).getPath());
		Boolean load = this.faceCascade.load(auxFile.getAbsolutePath());
		System.out.println("classifier :" + auxFile.getPath() + " loaded:" + load);
	}

	/**
	 * Detects faces on the input and returns a list of rectangulars around each
	 * detected face.
	 */
	public List<Rect> detectFace(final Mat frame) {
		Mat grayFrame = new Mat();
		float minHeight = 0.1f;

		Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

		// equalize the frame histogram to improve the result
		Imgproc.equalizeHist(grayFrame, grayFrame);

		// compute minimum face size (minHeight% of the frame height, in our case)

		int minSize = 0;
		int height = grayFrame.rows();

		if (Math.round(height * minHeight) > 0) {
			minSize = Math.round(height * minHeight);
		}

		final MatOfRect detectedFacesRectangulars = new MatOfRect();

		this.faceCascade.detectMultiScale(grayFrame, detectedFacesRectangulars, 1.05, 4, 0, new Size(minSize, minSize),
				grayFrame.size());

		return detectedFacesRectangulars.toList();
	}

}
