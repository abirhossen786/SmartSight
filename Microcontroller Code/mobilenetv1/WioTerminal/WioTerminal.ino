#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <MemoryFree.h>;
#include <pgmStrToRAM.h>; 
#include "int8.h"
#include "vw.h"


namespace {
// Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::ErrorReporter* reporter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 150*1024; // Just pick a big enough number
uint8_t tensor_arena[ kTensorArenaSize ] = { 0 };
float *input_buffer=nullptr;
}  // namespace



//void bitmap_to_float_array( float* dest, const unsigned char* bitmap ) { // Populate input_vec with the monochrome 1bpp bitmap
//  int pixel = 0;
//  for( int y = 0; y < 28; y++ ) {
//    for( int x = 0; x < 28; x++ ) {
//      int B = x / 8; // the Byte # of the row
//      int b = x % 8; // the Bit # of the Byte
//      dest[ pixel ] = ( bitmap[ y * 4 + B ] >> ( 7 - b ) ) & 
//                        0x1 ? 1.0f : 0.0f;
//      pixel++;
//    }
//  }
//}


void create_input( float *image ) {
  for (int x = 0; x < 27648; x++){
    input->data.f[x] = image[x];
  }
}


void setup() {
  // Load Model
  Serial.begin(115200);
  //delay(10000);
    while (!Serial) {
  }
  static tflite::MicroErrorReporter error_reporter;
  reporter = &error_reporter;
  //reporter->Report( "Let's use AI to recognize some numbers!" );

  model = tflite::GetModel( tf_model );
  if( model->version() != TFLITE_SCHEMA_VERSION ) {
    reporter->Report( "Model is schema version: %d\nSupported schema version is: %d", model->version(), TFLITE_SCHEMA_VERSION );
    return;
  }
  // Setup our TF runner
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, reporter );
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if( allocate_status != kTfLiteOk ) {
    reporter->Report( "AllocateTensors() failed" );
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Save the input buffer to put our MNIST images into
  input_buffer = input->data.f;
}

void loop() {

  ////////////////////////////////////////////////////////////////
// Pick a random test image for input
//  const int num_test_images = ( sizeof( test_images ) / sizeof( *test_images ) );


   create_input( (float *)seven );
   //bitmap_to_float_array( input_buffer, test_images[ rand() % num_test_images ] );

  // Run our model
  uint32_t start = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  if( invoke_status != kTfLiteOk ) {
    reporter->Report( "Invoke failed" );
    return;
  }
  
  float* result = output->data.f;
  uint32_t timeit = micros() - start;


//  for(int num =0; num <4608; num++){
//    Serial.println(result[num], 10);   
//    }


//    Serial.print("Free memory: ");
//    Serial.println(freeMemory());
//    Serial.print("MCU Inference time: ");
    Serial.println(timeit); 
//    Serial.println(" ms"); 
 
//for(;;){}
//delay(1000);

  
}
