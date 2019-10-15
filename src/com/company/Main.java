package com.company;

import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.protobuf.ByteString;
import com.google.protobuf.Int64Value;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import jdk.jshell.ImportSnippet;
import net.dongliu.requests.Requests;
import one.util.streamex.DoubleStreamEx;
import one.util.streamex.StreamEx;
import org.apache.commons.codec.binary.Base64;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;

import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_java;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.io.*;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.text.ParseException;
import java.util.*;

import static java.lang.Math.round;

public class Main {
    static {
            Loader.load(opencv_java.class);

        }




    public static void main(String[] args) throws IOException, ParseException {
        Mat mat=org.bytedeco.opencv.global.opencv_imgcodecs.imread("/home/danhyal/barcode.jpg");

        OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
        OpenCVFrameConverter.ToOrgOpenCvCoreMat converter2 = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();
        org.opencv.core.Mat converted=converter2.convert(converter1.convert(mat));
        org.opencv.core.Mat graymat=converter2.convert(converter1.convert(mat));
        org.opencv.imgproc.Imgproc.cvtColor(graymat,graymat, Imgproc.COLOR_RGBA2GRAY);
        org.opencv.core.Mat gradx=new org.opencv.core.Mat(graymat.width(),graymat.height(), CvType.CV_32F);
        org.opencv.core.Mat grady=new org.opencv.core.Mat(graymat.width(),graymat.height(), CvType.CV_32F);
        org.opencv.core.Mat finalm = new org.opencv.core.Mat(graymat.width(),graymat.height(), CvType.CV_32F);

        org.opencv.imgproc.Imgproc.Sobel(graymat,gradx,CvType.CV_32F,1,0,-1);
        org.opencv.imgproc.Imgproc.Sobel(graymat,grady,CvType.CV_32F,0,1,-1);

        org.opencv.core.Core.subtract(gradx,grady,finalm);
        org.opencv.core.Core.convertScaleAbs(finalm,finalm);
        org.opencv.imgproc.Imgproc.blur(finalm,finalm,new Size(9,9));
        org.opencv.imgproc.Imgproc.threshold(finalm,finalm,225,255, Imgproc.THRESH_BINARY);
        org.opencv.core.Mat struct=org.opencv.imgproc.Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(27,9));
        org.opencv.imgproc.Imgproc.morphologyEx(finalm,finalm,Imgproc.MORPH_CLOSE,struct);

        org.opencv.imgproc.Imgproc.erode(finalm,finalm,struct,new org.opencv.core.Point(0,0),5);
        Imgproc.dilate(finalm,finalm,struct,new org.opencv.core.Point(0,0),5);
        List<MatOfPoint> contours=new ArrayList<>();
        Imgproc.findContours(finalm,contours,new org.opencv.core.Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println(contours.size());
        List<org.opencv.core.Point> contour=contours.get(0).toList();
        Double contourarea=Imgproc.contourArea(contours.get(0));
        MatOfPoint2f point=new MatOfPoint2f();
        contours.get(0).convertTo(point,CvType.CV_32FC2);
        RotatedRect rect= Imgproc.minAreaRect(point);
        org.opencv.core.Mat boxpoints=new org.opencv.core.Mat();
        Imgproc.boxPoints(rect,boxpoints);


//        org.opencv.utils.Converters.Mat_to_vector_vector_Point(boxpoints,pts);
        System.out.println(rect.boundingRect());
//        Imgproc.drawContours(converted,pts,-1,new org.opencv.core.Scalar(0,255,0),3);
        Imgproc.rectangle(converted,rect.boundingRect(),new org.opencv.core.Scalar(0,255,0),3);
        org.opencv.imgcodecs.Imgcodecs.imwrite("/home/danhyal/grad.jpg",converted);



    }

    public void gsondetection() throws ParseException {
           OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
        OpenCVFrameConverter.ToOrgOpenCvCoreMat converter2 = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();


                     Mat img = org.bytedeco.opencv.global.opencv_imgcodecs.imread("/home/danhyal/notbroken.jpg");
                    while (true){
                                             final long startTime = System.currentTimeMillis();

                                             Mat test=jsonDetect(img);
                                             final long endTime = System.currentTimeMillis();
                                            System.out.println("Total execution time: " + (endTime - startTime));
                                            org.bytedeco.opencv.global.opencv_imgcodecs.imwrite("/home/danhyal/tff.jpg",test);

                    }




        }

        private static void grpcDetect(Mat img){
         String[] TensorCocoClasses=new String[]{
            "background",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "12",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "26",
            "backpack",
            "umbrella",
            "29",
            "30",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "45",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "66",
            "dining table",
            "68",
            "69",
            "toilet",
            "71",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "83",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"};
        long modelVersion = 1;
//          String host = "192.168.1.51";
        String host="92.233.63.88";
        int port = 8500;
        // the model's name.
        String modelName = "nasnet";

        // model's version
             org.bytedeco.opencv.global.opencv_imgproc.resize(img, img, new org.bytedeco.opencv.opencv_core.Size(1920, 1080));
//        ImageIO.write(image, "JPEG", out);
        ByteBuffer temp = img.getByteBuffer();
        byte[] arr = new byte[temp.remaining()];
        temp.get(arr);
        int[] options= new int[]{Imgcodecs.IMWRITE_JPEG_OPTIMIZE,1};
        opencv_imgcodecs.imencode(".jpg", img, arr);


        // create a channel
        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().enableFullStreamDecompression().keepAliveWithoutCalls(true).build();

        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel).withCompression("gzip").withMaxInboundMessageSize(402180070).withMaxOutboundMessageSize(402180070);

        // create a modelspec
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        modelSpecBuilder.setName(modelName);
        modelSpecBuilder.setVersion(Int64Value.of(modelVersion));
        modelSpecBuilder.setSignatureName("serving_default");

        Predict.PredictRequest.Builder builder = Predict.PredictRequest.newBuilder();
        builder.setModelSpec(modelSpecBuilder);

        // create the TensorProto and request
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        tensorProtoBuilder.setDtype(DataType.DT_STRING);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addStringVal(ByteString.copyFrom(arr));
        TensorProto tp = tensorProtoBuilder.build();

        builder.putInputs("inputs", tp);

        Predict.PredictRequest request = builder.build();
        Predict.PredictResponse response = stub.predict(request);

        Map<String, TensorProto> outputmap = response.getOutputsMap();
        int num_detections = (int) Objects.requireNonNull(outputmap.get("num_detections")).getFloatVal(0);
        List<Float> detection_classes = Objects.requireNonNull(outputmap.get("detection_classes")).getFloatValList();
        List<Float> detection_boxes_big = Objects.requireNonNull(outputmap.get("detection_boxes")).getFloatValList();
        List<List<Float>> detection_boxes = Lists.partition(detection_boxes_big, 4);
        List<Float> detection_scores= Objects.requireNonNull(outputmap.get("detection_scores")).getFloatValList();
        for (int j=0;j<num_detections;j+=1){
            double confidance=detection_scores.get(j);

            if (confidance>0.7){
                int top= (int) (detection_boxes.get(j).get(0)*img.rows());
                int left=(int)(detection_boxes.get(j).get(1)*img.cols());
                int bottom=(int)(detection_boxes.get(j).get(2)*img.rows());
                int right=(int)(detection_boxes.get(j).get(3)*img.cols());
                org.bytedeco.opencv.global.opencv_imgproc.rectangle(img,new Point(left,top),new Point(right,bottom), Scalar.GREEN);
                org.bytedeco.opencv.global.opencv_imgproc.putText(img,TensorCocoClasses[detection_classes.get(j).intValue()],new Point(left,top),1,0.5,Scalar.RED);


            }
        }


        channel.shutdown();
    }

        public static Mat jsonDetect(org.bytedeco.opencv.opencv_core.Mat mat) throws ParseException {
        Iterator i=null;
        Gson gson = new Gson();


            int time= (int) System.currentTimeMillis();
            ByteBuffer temp=mat.getByteBuffer();
            byte[] arr = new byte[temp.remaining()];
            temp.get(arr);
            opencv_imgcodecs.imencode(".jpg", mat, arr);
            String encoded= Base64.encodeBase64String(arr);
            org.json.simple.JSONObject json=new org.json.simple.JSONObject();
            org.json.simple.JSONArray oof=new org.json.simple.JSONArray();
            JSONObject b64=new JSONObject();

            b64.put("b64",encoded);oof.add(b64);json.put("instances",oof);
            String server_url = "http://localhost:8501/v1/models/nasnet:predict";
            JSONParser jsonParser=new JSONParser();
            String response = Requests.post(server_url).acceptCompress(true).keepAlive(true)
                    .jsonBody(json).socksTimeout(10000)
                    .send().readToText();
            try {
                    Object obj=jsonParser.parse(response);
            JSONObject jobj=(JSONObject) obj;
            JSONArray content=(JSONArray) jobj.get("predictions");
                assert content != null;
               i = content.iterator();

            }catch (Exception e){}


            JSONObject predictions = (JSONObject) i.next();
            int num_detections = ((Double) Objects.requireNonNull(predictions.get("num_detections"))).intValue();
            double[] detection_classes = gson.fromJson(Objects.requireNonNull(predictions.get("detection_classes")).toString(),(Type)double[].class);
            List<Double> detection_scores=new ArrayList<>();
            double[] detection_scoress =gson.fromJson(Objects.requireNonNull(predictions.get("detection_scores")).toString(),(Type)double[].class);
            for (double x:detection_scoress){ if (x!=0.0){ detection_scores.add(x); } }
            Double[][] detection_boxess=gson.fromJson(Objects.requireNonNull(predictions.get("detection_boxes")).toString(), (Type) Double[][].class);
            List<List<Double>> detection_boxes= StreamEx.of(detection_boxess).map(a -> DoubleStreamEx.of(a).boxed().toList()).toList();

            for (int j=0;j<num_detections;j+=1){
                double confidance=detection_scores.get(j);
                if (confidance>0.7){
                    int top= (int) (detection_boxes.get(j).get(0)*mat.rows());
                    int left=(int)(detection_boxes.get(j).get(1)*mat.cols());
                    int bottom=(int)(detection_boxes.get(j).get(2)*mat.rows());
                    int right=(int)(detection_boxes.get(j).get(3)*mat.cols());
                    org.bytedeco.opencv.global.opencv_imgproc.rectangle(mat,new Point(left,top),new Point(right,bottom), Scalar.GREEN);


                }

    //            int end= (int) (System.currentTimeMillis()-time);
    //        d(TAG, String.valueOf(end));
                }return mat;





    }

}
