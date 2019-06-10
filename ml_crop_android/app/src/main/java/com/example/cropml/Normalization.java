package com.example.cropml;


public class Normalization
{

    public static double min_max(double feature,double min,double max)
    {
        double new_min = 0;
        double new_max = 1;

        double normalize;
        normalize=(((feature-min)/(max-min))*(new_max-new_min))+new_min;

         return  normalize;
    }





}