import React, { useState, useRef } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { Button, TextInput, Card, Title, ActivityIndicator } from 'react-native-paper';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../App';
import { ApiService } from '../services/api';

type RegisterScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Register'>;

interface Props {
  navigation: RegisterScreenNavigationProp;
}

export default function RegisterScreen({ navigation }: Props) {
  const [name, setName] = useState('');
  const [permission, requestPermission] = useCameraPermissions();
  const [isCapturing, setIsCapturing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Card style={styles.card}>
          <Card.Content style={styles.cardContent}>
            <Title>Camera Permission Required</Title>
            <Button mode="contained" onPress={requestPermission} style={styles.button}>
              Grant Permission
            </Button>
          </Card.Content>
        </Card>
      </View>
    );
  }

  const handleCapture = async () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Please enter your name first');
      return;
    }

    if (!cameraRef.current) {
      Alert.alert('Error', 'Camera not ready');
      return;
    }

    setIsLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.7,
      });

      if (photo?.base64) {
        const result = await ApiService.registerUser(name.trim(), photo.base64);
        
        if (result.success) {
          Alert.alert('Success', 'Face registered successfully!', [
            { text: 'OK', onPress: () => navigation.goBack() }
          ]);
        } else {
          Alert.alert('Error', result.message || 'Registration failed');
        }
      }
    } catch (error) {
      console.error('Capture error:', error);
      Alert.alert('Error', 'Failed to capture image');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Card style={styles.inputCard}>
        <Card.Content>
          <TextInput
            label="Your Name"
            value={name}
            onChangeText={setName}
            mode="outlined"
            style={styles.input}
          />
        </Card.Content>
      </Card>

      <Card style={styles.cameraCard}>
        <Card.Content style={styles.cameraContent}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={CameraType.front}
          />
          <View style={styles.captureContainer}>
            <Button
              mode="contained"
              onPress={handleCapture}
              disabled={isLoading || !name.trim()}
              style={styles.captureButton}
              contentStyle={styles.buttonContent}
            >
              {isLoading ? <ActivityIndicator color="#fff" /> : 'Capture & Register'}
            </Button>
          </View>
        </Card.Content>
      </Card>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f4f6fa',
    padding: 20,
  },
  card: {
    elevation: 4,
    borderRadius: 12,
    marginBottom: 20,
  },
  inputCard: {
    marginTop: 20,
  },
  cameraCard: {
    flex: 1,
  },
  cardContent: {
    padding: 20,
    alignItems: 'center',
  },
  cameraContent: {
    flex: 1,
    padding: 16,
  },
  input: {
    marginBottom: 16,
  },
  camera: {
    flex: 1,
    borderRadius: 12,
    overflow: 'hidden',
  },
  captureContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  captureButton: {
    borderRadius: 8,
    minWidth: 200,
  },
  button: {
    borderRadius: 8,
    marginTop: 16,
  },
  buttonContent: {
    paddingVertical: 8,
  },
});</parameter>