package com.example.cropml;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.graphics.Typeface;
import android.os.Bundle;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import static java.util.Map.Entry.comparingByValue;
import static java.util.stream.Collectors.toMap;

public class MachineCrop extends AppCompatActivity implements View.OnClickListener {


    private String modelFile="crop_ml.tflite";
    Interpreter tflite;

    private EditText tempEdit;
    private  EditText humEdit;
    private Button pressButton;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_machine_crop);
        pressButton = (Button) findViewById(R.id.Mlbutton);
        tempEdit = (EditText) findViewById(R.id.tempeditText);
        humEdit = (EditText) findViewById(R.id.humideditText);
        pressButton.setOnClickListener(this);


    }

    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onClick(View v) {
        int view=v.getId();

        switch (view){

            case R.id.Mlbutton:
                try {
                    tflite=new Interpreter(loadModelFile(MachineCrop.this,modelFile));

                    String  humidity=humEdit.getText().toString().trim();
                    String temp =tempEdit.getText().toString().trim();

                    if (!validateForm(humEdit,tempEdit)) {
                        return;

                    }

                    HashMap<String, Float> predResult = new HashMap<String, Float>();


                    double humidity_norm=Normalization.min_max(Double.parseDouble(humidity),65,90);
                    double temp_norm=Normalization.min_max(Double.parseDouble(temp),13,29);

                    float[][] inp = {{(float) temp_norm, (float) humidity_norm}};
                    float[][] out=new float[][]{{0,0,0}};
                    tflite.run(inp,out);

                    float okroPropability=out[0][0]*100;
                    float onionPropability=out[0][1]*100;
                    float tomatoPropability=out[0][2]*100;

                    predResult.put("Okro", okroPropability);
                    predResult.put("Onion", onionPropability);
                    predResult.put("Tomato", tomatoPropability);

                    showDialog(MachineCrop.this,predResult);


                } catch (IOException e) {
                    e.printStackTrace();
                }

                break;


        }
    }


    public boolean validateForm(EditText humid, EditText temp ) {
        boolean valid = true;

        String hum = humid.getText().toString().trim();
        if (TextUtils.isEmpty(hum) || hum.length() <= 0 || hum.startsWith(" ")) {
            Toast toast=Toast.makeText(getApplicationContext(),"Provide Humidity Value",Toast.LENGTH_SHORT);
            View view=toast.getView();
            view.getBackground().setColorFilter(getResources().getColor(R.color.colorPrimary), PorterDuff.Mode.SRC_IN);
            TextView text = (TextView) view.findViewById(android.R.id.message);
            text.setShadowLayer(0, 0, 0, Color.TRANSPARENT);

            text.setTextColor(Color.WHITE);
            toast.show();
            valid = false;
        } else {
            humid.setError(null);
        }

        String tem = temp.getText().toString().trim();
        if (TextUtils.isEmpty(tem) || tem.length() <= 0 || tem.startsWith(" ")) {
            Toast toast= Toast.makeText(getApplicationContext(),"Provide Temp Value",Toast.LENGTH_SHORT);
            View  view=toast.getView();
            view.getBackground().setColorFilter(getResources().getColor(R.color.colorPrimary),PorterDuff.Mode.SRC_IN);
            TextView text = (TextView) view.findViewById(android.R.id.message);
            text.setShadowLayer(0, 0, 0, Color.TRANSPARENT);

            text.setTextColor(Color.WHITE);
            toast.show();
            valid = false;
        } else {
            temp.setError(null);
        }

        return valid;
    }


    public static HashMap<String, Float> sortByValue(HashMap<String, Float> hm)
    {
        // Create a list from elements of HashMap
        List<Map.Entry<String, Float> > list =
                new LinkedList<Map.Entry<String, Float> >(hm.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<String, Float> >() {
            public int compare(Map.Entry<String, Float> o1,
                               Map.Entry<String, Float> o2)
            {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        // put data from sorted list to hashmap
        HashMap<String, Float> temp = new LinkedHashMap<String, Float>();
        for (Map.Entry<String, Float> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }
        return temp;
    }

    private void showDialog(Context c,HashMap predictResult)  {

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(30,10,1,1);
        layout.setLayoutParams(new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.FILL_PARENT));


        predictResult=sortByValue(predictResult);
        Iterator<Map.Entry<String, Float>> itHash = predictResult.entrySet().iterator();


        boolean checker=false;
        while (itHash.hasNext()) {
            Map.Entry<String, Float> pair = (Map.Entry<String, Float>) itHash.next();
            final TextView input = new TextView(c);

            String key =pair.getKey();
            Float value = pair.getValue();
            input.setText( key + ": "+String.format("%.2f", value)+ "%");
            input.setPadding(20, 5, 5, 5);

            if (checker==false){
             input.setTypeface(input.getTypeface(), Typeface.BOLD);
             checker=true;

            }


            layout.addView(input);
        }



        AlertDialog dialog = new AlertDialog.Builder(c)
                .setTitle("Prediction Results")
                .setView(layout)
                .setNegativeButton("Ok", null)
                .create();


        dialog.show();

        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setTextColor(getResources().getColor(R.color.colorPrimary));
    }
}
