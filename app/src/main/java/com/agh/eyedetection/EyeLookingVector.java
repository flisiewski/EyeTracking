package com.agh.eyedetection;

import org.opencv.core.Point;

public class EyeLookingVector {

    Point iris;
    Point endPoint;
    double awayFromCenter;

    public EyeLookingVector(Point iris, Point endPoint, double awayFromCenter) {
        this.iris = iris;
        this.endPoint = endPoint;
        this.awayFromCenter = awayFromCenter;
    }

    public Point getIris() {
        return iris;
    }

    public void setIris(Point iris) {
        this.iris = iris;
    }

    public Point getEndPoint() {
        return endPoint;
    }

    public void setEndPoint(Point endPoint) {
        this.endPoint = endPoint;
    }

    public double getAwayFromCenter() {
        return awayFromCenter;
    }

    public void setAwayFromCenter(double awayFromCenter) {
        this.awayFromCenter = awayFromCenter;
    }
}
