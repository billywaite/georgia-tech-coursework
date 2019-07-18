package edu.gatech.cse6242;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapred.jobcontrol.JobControl;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import java.util.StringTokenizer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.mapreduce.lib.jobcontrol.ControlledJob;
import java.io.IOException;

public class Q4 {

  public static class DegreeMap extends Mapper<Object, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private final static IntWritable neg_one = new IntWritable(-1);

    private Text node = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      
			StringTokenizer token = new StringTokenizer(value.toString());

      while (token.hasMoreTokens()) {
        node.set(token.nextToken());
        context.write(node, one);
        node.set(token.nextToken());
        context.write(node, neg_one);
      }
    }
  }

  public static class DegreeReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
  	
		private IntWritable degree_reduced = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
      
			int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      degree_reduced.set(sum);
      context.write(key, degree_reduced);
    }
  }

  public static class DegreeCountMap extends Mapper<Object, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text degree = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      
			StringTokenizer token = new StringTokenizer(value.toString(), "\r");

      while (token.hasMoreTokens()) {
        String [] node = token.nextToken().split("\t");
        degree.set(node[1]);
        context.write(degree, one);

      }
    }
  }

  public static class DegreeCountReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
    
		private IntWritable diff_count = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
      
			int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      diff_count.set(sum);
      context.write(key, diff_count);
    }
  }

  public static void main(String[] args) throws Exception {
    
		Path temp = new Path("temp");
    
		Configuration conf = new Configuration();
    Job job1 = Job.getInstance(conf, "Q4");
    job1.setJarByClass(Q4.class);
    job1.setMapperClass(DegreeMap.class);
    job1.setCombinerClass(DegreeReducer.class);
    job1.setReducerClass(DegreeReducer.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job1, new Path(args[0]));
    FileOutputFormat.setOutputPath(job1, temp);
    boolean success = job1.waitForCompletion(true);

    if (success) {
      Configuration conf2 = new Configuration();
      Job job2 = Job.getInstance(conf2, "Q4");
      job2.setJarByClass(Q4.class);
      job2.setMapperClass(DegreeCountMap.class);
      job2.setCombinerClass(DegreeCountReducer.class);
      job2.setReducerClass(DegreeCountReducer.class);
      job2.setOutputKeyClass(Text.class);
      job2.setOutputValueClass(IntWritable.class);
      FileInputFormat.addInputPath(job2, temp);
      FileOutputFormat.setOutputPath(job2, new Path(args[1]));
      System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }

  }
}
