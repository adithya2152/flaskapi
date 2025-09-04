import React, { useState, useRef } from 'react';
import { View, StyleSheet, Alert } from 'react-native';
import { Button, Card, Title, ActivityIndicator, Text } from 'react-native-paper';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../../App';
import { ApiService } from '../services/api';
import AsyncStorage from '@react-native-async-storage/async-storage';

type LoginScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Login'>;

interface Props {
  navigation: LoginScreenNavigationProp;
}

export default function LoginScreen({ navigation }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
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

  const handleLogin = async () => {
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
        const result = await ApiService.loginUser(photo.base64);
        
        if (result.success && result.data?.username) {
          await AsyncStorage.setItem('username', result.data.username);
          navigation.navigate('Monitoring', { username: result.data.username });
        } else {
          Alert.alert('Login Failed', result.message || 'Face not recognized');
        }
      }
    } catch (error) {
      console.error('Login error:', error);
      Alert.alert('Error', 'Failed to capture image for login');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Card style={styles.instructionCard}>
        <Card.Content>
          <Text style={styles.instruction}>
            Position your face in the camera and tap "Login with Face" to authenticate
          </Text>
        </Card.Content>
      </Card>

      <Card style={styles.cameraCard}>
        <Card.Content style={styles.cameraContent}>
          <CameraView
            ref={cameraRef}
            style={styles.camera}
            facing={CameraType.front}
          />
          <View style={styles.loginContainer}>
            <Button
              mode="contained"
              onPress={handleLogin}
              disabled={isLoading}
              style={styles.loginButton}
              contentStyle={styles.buttonContent}
            >
              {isLoading ? <ActivityIndicator color="#fff" /> : 'Login with Face'}
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
  instructionCard: {
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
  instruction: {
    textAlign: 'center',
    fontSize: 16,
    color: '#555',
  },
  camera: {
    flex: 1,
    borderRadius: 12,
    overflow: 'hidden',
  },
  loginContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  loginButton: {
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